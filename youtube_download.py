import yt_dlp

url = "https://www.youtube.com/watch?v=1StmDwRpwkQ"
opts = {
    "format": "bestvideo+bestaudio/best",
    "merge_output_format": "mp4"
}
with yt_dlp.YoutubeDL(opts) as ydl:
    ydl.download([url])


