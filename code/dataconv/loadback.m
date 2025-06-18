% Load back the data and found ROIs on gaon
function [dFF, rois, rnear] = loadback(idx)
    files=sort(split(ls('SEUDOdata/FullData/')));
    % remove the 0x0 element
    files = files(2:end);
	nf = size(files, 1);

    f = mtt(files, idx, 1);
    disp(f);
    load(f);
    load("SEUDOresults/rois_" + f);

    rois_event("set", evs);
end