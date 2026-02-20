```shell
ffmpeg -i /Users/yuli/Desktop/remecli_en.mov \
       -vf "scale=-2:1080,setpts=0.333*PTS" \
       -c:v libx264 \
       -crf 28 \
       -preset fast \
       -c:a aac \
       -b:a 96k \
       /Users/yuli/Desktop/remecli_en_1080p_3x.mp4

ffmpeg -i /Users/yuli/Desktop/remecli_zh.mov \
       -vf "scale=-2:1080,setpts=0.333*PTS" \
       -c:v libx264 \
       -crf 28 \
       -preset fast \
       -c:a aac \
       -b:a 96k \
       /Users/yuli/Desktop/remecli_zh_1080p_3x.mp4
```