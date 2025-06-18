% Load all SEUDO data files and convert them to tiff format, one by one
%
% from - index of the first file to process (starting from 1)
% to - index of the last file to process, or -1 to the end
function convert_all_to_tiff(from, to)
	files = list_data_files();

	nf = size(files, 1);
	if to < 0 || to > nf
		to = nf;
	end

	tiff_dir = "SEUDOtiff/";

	for i=from:to
		f = files(i);
		disp(f);

		load(f + ".mat");

		outf = tiff_dir + replace(f, "-", "_") + ".tiff";
		disp(outf);
		export_as_tiff(dFF, outf);
	end
end
