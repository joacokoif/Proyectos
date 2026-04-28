import tkinter as tk
from tkinter import font
import asyncio
import threading
import queue
from media_tracker import get_current_media_info, get_current_position
from lyrics_fetcher import fetch_lyrics

class LyricItem:
    def __init__(self, idx, canvas, text, bold_font, dim_font, width):
        self.idx = idx
        self.canvas = canvas
        self.text = text if text.strip() else "♪"
        self.bold_font = bold_font
        self.dim_font = dim_font
        self.width = width
        
        # Initial draw off-screen
        self.id = canvas.create_text(0, -1000, text=self.text, font=dim_font, fill="#60646b", anchor="nw", width=width)
        
        self.base_y = 0.0
        self.height = 30
        self.target_y = 0.0
        self.visual_y = -1000.0  # start out of view
        
        self.is_active = False
        self.recalc_height()

    def set_active(self, active):
        if self.is_active != active:
            self.is_active = active
            if active:
                self.canvas.itemconfig(self.id, font=self.bold_font, fill="white")
            else:
                self.canvas.itemconfig(self.id, font=self.dim_font, fill="#60646b")
            self.recalc_height()

    def set_width(self, new_width):
        if new_width != self.width:
            self.width = new_width
            self.canvas.itemconfig(self.id, width=new_width)
            self.recalc_height()

    def recalc_height(self):
        bbox = self.canvas.bbox(self.id)
        self.height = (bbox[3] - bbox[1]) if bbox else 30

    def update_physics(self, target_scroll_y, is_instant=False):
        # Target position is its sequential layout position + global scroll offset
        self.target_y = self.base_y + target_scroll_y
        
        if is_instant:
            self.visual_y = self.target_y
        
        diff = self.target_y - self.visual_y
        if abs(diff) > 0.5:
            # Ease towards the target (Spotify flow)
            self.visual_y += diff * 0.15
        else:
            self.visual_y = self.target_y
            
        self.canvas.coords(self.id, 0, self.visual_y)

class SpotifyLyricsWidget:
    def __init__(self, root):
        self.root = root
        self.width = 650
        self.height = 200
        
        x_pos = (root.winfo_screenwidth() // 2) - (self.width // 2)
        y_pos = root.winfo_screenheight() - self.height - 150
        
        self.root.geometry(f"{self.width}x{self.height}+{x_pos}+{y_pos}")
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        
        transparent_color = "black"
        self.root.configure(bg=transparent_color)
        self.root.wm_attributes("-transparentcolor", transparent_color)
        self.root.attributes("-alpha", 0.95)
        
        self.canvas = tk.Canvas(root, bg=transparent_color, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.bg_polygon = self.draw_rounded_rect(0, 0, self.width-1, self.height-1, 20, fill="#2b2d31", outline="#4a4d52", width=1)
        
        self.canvas.bind("<ButtonPress-1>", self.start_move)
        self.canvas.bind("<ButtonRelease-1>", self.stop_move)
        self.canvas.bind("<B1-Motion>", self.do_move)

        # Fonts
        self.bold_font = font.Font(family="Segoe UI", size=22, weight="bold")
        self.dim_font = font.Font(family="Segoe UI", size=16)

        # Header
        self.canvas.create_oval(20, 20, 30, 30, fill="#1DB954", outline="")
        self.canvas.create_text(40, 25, text="Spotify Subtitles", fill="#e3e5e8", font=("Segoe UI", 10, "bold"), anchor="w")
        
        self.close_btn = self.canvas.create_oval(self.width-35, 15, self.width-15, 35, fill="#3f4248", outline="")
        self.close_text = self.canvas.create_text(self.width-25, 25, text="✕", fill="#b5bac1", font=("Segoe UI", 10, "bold"))
        self.canvas.tag_bind(self.close_btn, "<Button-1>", lambda e: self.root.destroy())
        self.canvas.tag_bind(self.close_text, "<Button-1>", lambda e: self.root.destroy())

        # Resize grip
        self.grip = self.canvas.create_polygon(self.width-15, self.height-5, self.width-5, self.height-5, self.width-5, self.height-15, fill="#80848a", outline="")
        self.canvas.tag_bind(self.grip, "<ButtonPress-1>", self.start_resize)
        self.canvas.tag_bind(self.grip, "<ButtonRelease-1>", self.stop_resize)
        self.canvas.tag_bind(self.grip, "<B1-Motion>", self.do_resize)
        self.canvas.tag_bind(self.grip, "<Enter>", lambda e: self.canvas.config(cursor="size_nw_se"))
        self.canvas.tag_bind(self.grip, "<Leave>", lambda e: self.canvas.config(cursor=""))

        # Bottom song info area
        self.song_info_id = self.canvas.create_text(20, self.height-25, text="No song playing", fill="#80848a", font=("Segoe UI", 10), anchor="nw")

        # Internal canvas for lyrics clipping. Note: height starts lower to clear top, and ends higher to clear bottom.
        self.lyrics_canvas = tk.Canvas(self.canvas, bg="#2b2d31", highlightthickness=0)
        self.lyrics_window_id = self.canvas.create_window(20, 45, window=self.lyrics_canvas, anchor="nw", 
                                                          width=self.width-40, height=self.height-85)
        
        self.lyrics_canvas.bind("<ButtonPress-1>", self.start_move)
        self.lyrics_canvas.bind("<ButtonRelease-1>", self.stop_move)
        self.lyrics_canvas.bind("<B1-Motion>", self.do_move)

        # State initialization
        self.current_song = None
        self.song_album_info = ""
        self.queue = queue.Queue()
        
        self.lyric_items = []
        self.raw_parsed_lyrics = []
        self.current_line_idx = -1
        self.target_scroll_y = 0.0
        
        self.sync_offset = 0.1

        self._drag_start_x = None
        self._drag_start_y = None
        self.is_resizing = False

        self.thread = threading.Thread(target=self.run_async_loop, daemon=True)
        self.thread.start()

        self.root.after(100, self.check_queue)
        self.animate_physics()

    def start_move(self, event):
        self._drag_start_x = event.x_root
        self._drag_start_y = event.y_root

    def stop_move(self, event):
        self._drag_start_x = None
        self._drag_start_y = None

    def do_move(self, event):
        if self.is_resizing: return
        if getattr(self, '_drag_start_x', None) is None: return
        try:
            deltax = event.x_root - self._drag_start_x
            deltay = event.y_root - self._drag_start_y
            x = self.root.winfo_x() + deltax
            y = self.root.winfo_y() + deltay
            self.root.geometry(f"+{x}+{y}")
            
            self._drag_start_x = event.x_root
            self._drag_start_y = event.y_root
        except: pass

    def start_resize(self, event):
        self.is_resizing = True
        self._resize_start_x = event.x_root
        self._resize_start_y = event.y_root
        self._resize_start_w = self.width
        self._resize_start_h = self.height

    def stop_resize(self, event):
        self.is_resizing = False

    def do_resize(self, event):
        if getattr(self, '_resize_start_x', None) is None: return
        
        deltax = event.x_root - self._resize_start_x
        deltay = event.y_root - self._resize_start_y
        
        new_width = max(250, self._resize_start_w + deltax)
        new_height = max(130, self._resize_start_h + deltay)
        
        if self.width != new_width or self.height != new_height:
            self.width = new_width
            self.height = new_height
            self.root.geometry(f"{self.width}x{self.height}")
            self.redraw_ui()

    def get_rounded_rect_points(self, x1, y1, x2, y2, r):
        return [x1+r, y1, x1+r, y1, x2-r, y1, x2-r, y1, x2, y1, x2, y1+r, x2, y1+r, x2, y2-r, x2, y2-r, x2, y2, x2-r, y2, x2-r, y2, x1+r, y2, x1+r, y2, x1, y2, x1, y2-r, x1, y2-r, x1, y1+r, x1, y1+r, x1, y1]

    def draw_rounded_rect(self, x1, y1, x2, y2, r, **kwargs):
        points = self.get_rounded_rect_points(x1, y1, x2, y2, r)
        return self.canvas.create_polygon(points, **kwargs, smooth=True)

    def redraw_ui(self):
        # Update background
        points = self.get_rounded_rect_points(0, 0, self.width-1, self.height-1, 20)
        self.canvas.coords(self.bg_polygon, *points)
        
        self.canvas.coords(self.close_btn, self.width-35, 15, self.width-15, 35)
        self.canvas.coords(self.close_text, self.width-25, 25)
        
        self.canvas.coords(self.song_info_id, 20, self.height-25)
        self.update_song_info()
        
        self.canvas.coords(self.grip, self.width-15, self.height-5, self.width-5, self.height-5, self.width-5, self.height-15)
        
        # Reposition lyrics canvas
        self.canvas.itemconfig(self.lyrics_window_id, width=self.width-40, height=max(1, self.height-85))
        
        # Tell all lyrics to re-wrap and recalculate heights instantly
        if self.lyric_items:
            for item in self.lyric_items:
                item.set_width(self.width - 40)
            self.recalc_layout()

    def run_async_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.background_task())

    async def background_task(self):
        while True:
            try:
                media_info = await get_current_media_info()
                if media_info:
                    song_id = f"{media_info['title']} - {media_info['artist']}"
                    if song_id != self.current_song:
                        self.current_song = song_id
                        
                        self.queue.put({
                            "type": "update_song",
                            "info": {
                                "title": media_info['title'],
                                "artist": media_info['artist'],
                                "album": media_info.get('album', '')
                            }
                        })
                        
                        self.queue.put({
                            "type": "update_lyrics_status",
                            "text": "Fetching lyrics..."
                        })

                        lyrics_data = await fetch_lyrics(media_info['title'], media_info['artist'])
                        if lyrics_data and lyrics_data.get('parsed'):
                            self.queue.put({
                                "type": "update_lyrics",
                                "parsed_lyrics": lyrics_data['parsed']
                            })
                        else:
                            self.queue.put({
                                "type": "update_lyrics_status",
                                "text": "Lyrics not found for this song."
                            })
                            
                if self.current_song:
                    pos = await get_current_position()
                    if pos > 0:
                        self.queue.put({"type": "update_position", "position": pos})
                        
                else:
                    if self.current_song is not None:
                        self.current_song = None
                        self.queue.put({"type": "update_song", "info": None})
                        self.queue.put({"type": "update_lyrics_status", "text": "Play a song on Spotify to see lyrics here."})
                    
            except Exception as e:
                pass 
            
            await asyncio.sleep(0.1)

    def check_queue(self):
        try:
            while True:
                msg = self.queue.get_nowait()
                msg_type = msg.get("type")
                
                if msg_type == "update_song":
                    self.update_song_info(msg["info"])
                elif msg_type == "update_lyrics_status":
                    self.load_status_text(msg["text"])
                elif msg_type == "update_lyrics":
                    self.load_lyrics_data(msg["parsed_lyrics"])
                elif msg_type == "update_position":
                    if self.lyric_items:
                        self.sync_lyrics(msg["position"])
        except queue.Empty:
            pass
        finally:
            self.root.after(50, self.check_queue) 

    def update_song_info(self, info_dict=False):
        if info_dict is not False:
             self.current_song_data = info_dict
             
        if getattr(self, 'current_song_data', None) is None:
             self.canvas.itemconfig(self.song_info_id, text="No song playing")
             return

        title = self.current_song_data['title']
        artist = self.current_song_data['artist']
        album = self.current_song_data['album']
        
        level3 = f"{title} • {artist} • {album}" if album else f"{title} • {artist}"
        level2 = f"{title} • {artist}"
        level1 = f"{title}"
        
        self.canvas.itemconfig(self.song_info_id, text=level3)
        bbox = self.canvas.bbox(self.song_info_id)
        if bbox and (bbox[2] - bbox[0]) > (self.width - 40):
            self.canvas.itemconfig(self.song_info_id, text=level2)
            bbox = self.canvas.bbox(self.song_info_id)
            if bbox and (bbox[2] - bbox[0]) > (self.width - 40):
                self.canvas.itemconfig(self.song_info_id, text=level1)
                bbox = self.canvas.bbox(self.song_info_id)
                # Hard truncate if still too long
                text = level1
                while len(text) > 3 and bbox and (bbox[2] - bbox[0]) > (self.width - 40):
                    text = text[:-2] + "…"
                    self.canvas.itemconfig(self.song_info_id, text=text)
                    bbox = self.canvas.bbox(self.song_info_id)

    def load_status_text(self, text):
        for item in self.lyric_items: self.lyrics_canvas.delete(item.id)
        self.lyric_items = []
        self.lyrics_canvas.delete("all")
        
        status_item = LyricItem(0, self.lyrics_canvas, text, self.bold_font, self.bold_font, self.width-40)
        status_item.is_active = True
        status_item.canvas.itemconfig(status_item.id, fill="#e3e5e8")
        
        self.lyric_items.append(status_item)
        self.current_line_idx = 0
        self.recalc_layout(instant=True)

    def load_lyrics_data(self, parsed_lyrics):
        self.raw_parsed_lyrics = parsed_lyrics
        for item in self.lyric_items: self.lyrics_canvas.delete(item.id)
        self.lyric_items = []
        
        for i, lyric in enumerate(parsed_lyrics):
            item = LyricItem(i, self.lyrics_canvas, lyric["text"], self.bold_font, self.dim_font, self.width-40)
            self.lyric_items.append(item)
            
        self.current_line_idx = -1
        self.recalc_layout(instant=True)

    def recalc_layout(self, instant=False):
        # 1. Update cascade positions
        current_y = 0.0
        for item in self.lyric_items:
            item.base_y = current_y
            current_y += item.height + 15
            
        # 2. Update scroll target to center the active item (or first item if none)
        idx = max(0, self.current_line_idx)
        if idx < len(self.lyric_items):
            active_item = self.lyric_items[idx]
            viewport_h = max(1, self.height - 85)
            self.target_scroll_y = (viewport_h / 2) - (active_item.base_y + active_item.height / 2)
        
        if instant:
            for item in self.lyric_items:
                item.update_physics(self.target_scroll_y, is_instant=True)

    def animate_physics(self):
        # 60fps tick
        for item in self.lyric_items:
            item.update_physics(self.target_scroll_y)
        self.root.after(16, self.animate_physics)

    def sync_lyrics(self, position):
        new_idx = -1
        adjusted_position = position + self.sync_offset
        
        for i, lyric in enumerate(self.raw_parsed_lyrics):
            if lyric["time"] <= adjusted_position:
                new_idx = i
            else:
                break
                
        if new_idx != self.current_line_idx and new_idx != -1 and new_idx < len(self.lyric_items):
            self.current_line_idx = new_idx
            
            # Apply styling
            for i, item in enumerate(self.lyric_items):
                item.set_active(i == new_idx)
                
            # Triggers a recalculation which adjusts Heights and target Global Scroll
            self.recalc_layout()

if __name__ == "__main__":
    root = tk.Tk()
    app = SpotifyLyricsWidget(root)
    root.mainloop()
