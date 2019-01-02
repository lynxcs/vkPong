#!/bin/bash
cd /home/void/devel/vulkan/vkPong/textures
for i in `seq 0 9`; do
    let charCode=48+i
    msdfgen msdf -font /usr/share/fonts/noto/NotoSerif-Black.ttf $charCode -o font_$i.png -size 32 32 -pxrange 4 -autoframe
done
