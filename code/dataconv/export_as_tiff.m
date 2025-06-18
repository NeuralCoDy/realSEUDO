% Export the movie in SEUDO format (X x Y x Frames) into a TIFF file.
%
% M - the movie, a 3-dimensional array with dimensions X, Y, Frame
% filename - file to export to
function export_as_tiff(M, filename)
	% pixel values will have this added
	offset = 10;

	% show those, for information
	maxval = max(M, [], "all");
	minval = min(M, [], "all");

	% pixel values will be multiplied by this
	if maxval == minval
		scale = 1;
	else
		scale = 5000 / (maxval - minval);
	end;

	t = Tiff(filename, 'w');

	tags.ImageLength = size(M, 1);
	tags.ImageWidth = size(M, 2);
	tags.BitsPerSample = 16;
	tags.SamplesPerPixel = 1;
	tags.Compression = Tiff.Compression.AdobeDeflate;
	tags.Photometric = Tiff.Photometric.MinIsBlack;
	tags.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;

	for i = 1:size(M, 3)
		setTag(t, tags)
		write(t, uint16((M(:,:,i) - minval) * scale + offset));
		writeDirectory(t);
	end

	close(t);
end

% Export a SEUDO data set in TIFF format, as multiple TOFF files
% in a directory. They can be later concatenated into one file.
function export_as_tiff_old(M, dstbase, fragsz)
	nframes = size(M, 3);
	nfrags = ceil(nframes / fragsz);

	% pixel values will be multiplied by this
	scale = 100;
	% pixel values will have this added
	offset = 500;

	for frag=1:nfrags
		start = (frag - 1) * fragsz;
		outfile = sprintf("%s_%07d.tiff", dstbase, frag);
		disp(outfile);

		img = M(:,:, start + 1);
		imwrite(uint16(img .* scale + offset), outfile);

		last = start + fragsz;
		if last > nframes
			last = nframes;
		end

		for f = (start+2):last
		    img = M(:,:, f);
		    imwrite(uint16(img .* scale + offset), outfile, "WriteMode","append");
		end
	end
end
