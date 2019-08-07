clc
clear
video_file = VideoWriter('ga.avi');
video_file.FrameRate = 1;
open(video_file);

imageDir = './figs/';
files=dir(imageDir);
files(1:2)=[];
for file = files'
   img = imread([imageDir file.name]);
   writeVideo(video_file,img)
end
close(video_file);