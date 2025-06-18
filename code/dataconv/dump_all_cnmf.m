% Dump all the results from the preocessed CNMF files into graphs
%
% from - index of the first file to process (starting from 1)
% to - index of the last file to process, or -1 to the end
function dump_all_cnmf(from, to)
	files = list_data_files();

	nf = size(files, 1);
	if to < 1 || to > nf
		to = nf;
	end

	res_dir = "CaImAn/example_movies/SEUDOtiff/";
	trace_prefix = res_dir + "cnmf_trace_";
	outl_prefix = res_dir + "cnmf_outlines_";
	show_prefix = res_dir + "cnmf_show_";

	for i=from:to
		f = files(i);
		disp(f);

		% Matlab doesn't allow "-" in the file names, so made symlinks with "_" instead
		normf = replace(f, '-', '_');

		load(f + ".mat");
		run(normf + "_cnmf_shapes");
		run(normf + "_cnmf_traces");
		rois = rois_from_plain(shapes);

		hmap = rois_heatmap(dFF);
		show_trace(traces, 1, 'rois', rois, 'heatmap', hmap, 'filename', char(trace_prefix + f + ".png"));
		show_trace(traces, 1, 'rois', rois, 'heatmap', hmap, 'filename', char(trace_prefix + f + ".pdf"));
		show_rois_outlines(hmap, rois, 4, char(outl_prefix + f + ".png"));
		show_rois(rois, 2, char(show_prefix + f + ".png"));
	end
end
