% List all the data files on gaon server, removing the .mat suffix.
%
function result = list_data_files()
	files=sort(split(ls('SEUDOdata/FullData/')));
	% remove the 0x0 element by skipping index 1
	result = replace(string(files(2:end)), ".mat", "");
end
