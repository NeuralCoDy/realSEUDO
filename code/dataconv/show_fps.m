
if 0 % {
	% early version, without error bars
	tbl_seudo = readtable('figures-gaon/seudo3/summary_seudo.csv');
	tbl_onacid = readtable('figures-gaon/onacid2/summary_onacid.csv');

	f = figure();
	ax = axes();
	% s = scatter(ax, [tbl_seudo.(1); tbl_onacid.(1)], [tbl_seudo.(2); tbl_onacid.(2)], "filled");
	% s.MarkerFaceAlpha = 0.5;
	% s.CData = [ 0.1, 0.1, 1; 0.1, 1, 0.1];
	% s.DisplayName = ["SEUDO", "OnACID"];

	ss = scatter(ax, tbl_seudo.(1), tbl_seudo.(2), "filled","bo");
	ss.DisplayName = "SEUDO";

	hold on;
	so = scatter(ax, tbl_onacid.(1), tbl_onacid.(2), "filled","gv");
	so.DisplayName = "OnACID";

	ax.XLabel.String = "Detected ROIs In Movie";
	ax.YLabel.String = "Frames Per Second";

	legend(ax);

	f = figure();
	ax = axes();

	ss = scatter(ax, tbl_seudo.(1), tbl_seudo.(5) ./ tbl_seudo.(2) / 100, "filled","bo");
	ss.DisplayName = "SEUDO";

	hold on;
	so = scatter(ax, tbl_onacid.(1), tbl_onacid.(5) ./ tbl_onacid.(2) / 100, "filled","gv");
	so.DisplayName = "OnACID";

	ax.XLabel.String = "Detected ROIs In Movie";
	ax.YLabel.String = "CPU Seconds Per Frame";

	legend(ax);
end % }

% with error bars on SEUDO

% result from:
%   perl ../../dataconv/parse_logs.pl SE*.log >summary2_seudo.csv
tbl_seudo = readtable('figures-gaon/seudo4/summary2_seudo.csv');

% result from:
%   for i in *.log; do f=`basename $i .log`; echo $f; wc -l ${f}_onacid_traces.m >> $i; done
%   perl ../../dataconv/parse_onacid_logs.pl SE*.log >summary2_onacid.csv
tbl_onacid = readtable('figures-gaon/onacid4/summary2_onacid.csv');

% Figure: FPS from ROI count

f = figure();
ax = axes();
% s = scatter(ax, [tbl_seudo.(1); tbl_onacid.(1)], [tbl_seudo.(2); tbl_onacid.(2)], "filled");
% s.MarkerFaceAlpha = 0.5;
% s.CData = [ 0.1, 0.1, 1; 0.1, 1, 0.1];
% s.DisplayName = ["SEUDO", "OnACID"];

ss = errorbar(ax, tbl_seudo.(1), tbl_seudo.(2), tbl_seudo.(3), "bo");
ss.DisplayName = "SEUDO";

hold on;
so = errorbar(ax, tbl_onacid.(1), tbl_onacid.(2), tbl_onacid.(3), "gv");
so.DisplayName = "OnACID";

ax.XLabel.String = "Detected ROIs In Movie";
ax.YLabel.String = "Frames Per Second";

legend(ax);

% Figure: CPU seconds per frame from ROI count

f = figure();
ax = axes();

ss = errorbar(ax, tbl_seudo.(1), tbl_seudo.(4), tbl_seudo.(5), "bo");
ss.DisplayName = "SEUDO";

hold on;
so = errorbar(ax, tbl_onacid.(1), tbl_onacid.(4), tbl_onacid.(5), "gv");
so.DisplayName = "OnACID";

ax.XLabel.String = "Detected ROIs In Movie";
ax.YLabel.String = "CPU Seconds Per Frame";

legend(ax);
