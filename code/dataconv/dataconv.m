% convert SEUDO data from M, loaded with
%    load demoData1.mat
% to a multi-page 16-bit TIFF

% print the min and max values
%vmin = min(reshape(M, 1, []));
%vmax = max(reshape(M, 1, []));
%vmin
%vmax

% pixel values will be multiplied by this
scale = 100;
% pixel values will have this added
offset = 500;

outfile = "/home/babkin/w/yulia/dataconv/data2.tiff";

% frames = [1:size(M, 3)];
frames = [1:2000];
img = M(:,:, frames(1));
imwrite(uint16(img .* scale + offset), outfile);

for f = frames(2:end)
    disp(f);
    img = M(:,:, f);
    imwrite(uint16(img .* scale + offset), outfile, "WriteMode","append");
end

%t = Tiff('two.tiff', 'w');
%tags.ImageLength = size(M, 1);
%tags.ImageWidth = size(M, 2);
%%tags.TileLength = size(M, 1);
%%tags.TileWidth = size(M, 2);
%tags.BitsPerSample = 16;
%tags.SamplesPerPixel = 1;
%tags.Compression = Tiff.Compression.AdobeDeflate;
%tags.Photometric = Tiff.Photometric.MinIsBlack;
%tags.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
%%tags.SubFileType = Tiff.SubFileType.Page;
%
%setTag(t, tags)
%write(t, uint16(M(:,:,1) * 100 + 500));
%writeDirectory(t);
%
%close(t);
