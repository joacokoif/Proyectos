import syncedlyrics
import asyncio
import re

def parse_lrc(lrc_text):
    """
    Parses LRC text into a list of dictionaries with time (in seconds) and text.
    """
    if not lrc_text:
        return []

    lines = lrc_text.strip().split('\n')
    parsed_lyrics = []
    
    # regex for [mm:ss.xx] or [mm:ss]
    time_prog = re.compile(r'\[(\d+):(\d+\.?\d*)\]')
    
    for line in lines:
        match = time_prog.match(line)
        if match:
            minutes = int(match.group(1))
            seconds = float(match.group(2))
            total_seconds = minutes * 60 + seconds
            text = line[match.end():].strip()
            parsed_lyrics.append({"time": total_seconds, "text": text})
            
    return parsed_lyrics

async def fetch_lyrics(title, artist):
    """
    Searches for synchronized lyrics using syncedlyrics and returns parsed LRC.
    """
    search_term = f"{title} {artist}"
    loop = asyncio.get_event_loop()
    
    # syncedlyrics.search is synchronous, run it in an executor
    lrc_text = await loop.run_in_executor(None, syncedlyrics.search, search_term)
    
    if lrc_text:
        return {
            "raw": lrc_text,
            "parsed": parse_lrc(lrc_text)
        }
    return None
