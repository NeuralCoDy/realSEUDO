% Dump all the results from the preocessed SEUDO files into graphs
%
% from - index of the first file to process (starting from 1)
% to - index of the last file to process, or -1 to the end
function dump_all_results(from, to)
	files = list_data_files();

	nf = size(files, 1);
	if to < 0 || to > nf
		to = nf;
	end

	res_dir = "SEUDOresults/";
	res_prefix = res_dir + "rois_";
	trace_prefix = res_dir + "trace_";
	outl_prefix = res_dir + "outlines_";
	show_prefix = res_dir + "show_";

	for i=from:to
		f = files(i);
		disp(f);

		[M, rois, rnear] = loadback(i);
		hmap = rois_heatmap(M);
		show_rois_trace(rois, 1, 'heatmap', hmap, 'filename', char(trace_prefix + f + ".png"));
		show_rois_trace(rois, 1, 'heatmap', hmap, 'filename', char(trace_prefix + f + ".pdf"));
		show_rois_outlines(hmap, rois, 4, char(outl_prefix + f + ".png"));
		show_rois(rois, 2, char(show_prefix + f + ".png"));
	end
end
