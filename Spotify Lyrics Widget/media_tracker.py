import asyncio
import datetime
from winsdk.windows.media.control import GlobalSystemMediaTransportControlsSessionManager as MediaManager

class MediaTracker:
    def __init__(self):
        self.manager = None

    async def _get_manager(self):
        if not self.manager:
            self.manager = await MediaManager.request_async()
        return self.manager

    async def get_current_media_info(self):
        """
        Retrieves the currently playing media info from Windows GSMTC.
        Returns a dictionary with 'title', 'artist', and 'album'.
        """
        manager = await self._get_manager()
        current_session = manager.get_current_session()
        
        if current_session:
            info = await current_session.try_get_media_properties_async()
            return {
                "title": info.title,
                "artist": info.artist,
                "album": info.album_title
            }
        return None

    async def get_current_position(self):
        """
        Returns the current playback position in seconds (float).
        """
        manager = await self._get_manager()
        current_session = manager.get_current_session()
        
        if current_session:
            try:
                timeline = current_session.get_timeline_properties()
                playback = current_session.get_playback_info()
                if timeline:
                    base_pos = timeline.position.total_seconds()
                    
                    # winsdk caches position, so we MUST interpolate it using system time!
                    last_updated = timeline.last_updated_time
                    if last_updated and playback and playback.playback_status == 4: # 4 = Playing
                        # Calculate elapsed time since last update
                        now = datetime.datetime.now(datetime.timezone.utc)
                        elapsed = (now - last_updated).total_seconds()
                        return base_pos + elapsed
                    else:
                        return base_pos
            except Exception:
                pass
        return 0.0

tracker = MediaTracker()
get_current_media_info = tracker.get_current_media_info
get_current_position = tracker.get_current_position

