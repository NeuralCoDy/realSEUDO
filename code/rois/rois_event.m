
% Record and extract the events of ROI processing.
%
% Args:
% opcode - operation in the event:
%   clear - clear all the events;
%     the optional next argument is a flag for whether the native implmentation
%     should be enabled
%   get - get all the events;
%   get_trace - get the processed trace for one ROI
%   get_last_frame - get the frame of the last event
%   set - set back the events previously extracted by "get";
%   detect - detected a brightness of a ROI;
%   detect_early - detected a brightness of a potential ROI, these values are less precise;
%   mod_merge - two ROIs got merged;
%   mod_subtract - two ROIs got subtracted;
%   mod_shard - two ROIs got subtracted, and that subtraction produced extra shards.
% The rest in varargin are skipped for opcodes "clear" and "get";
% for opcode "set" contain a single argument of cell array;
% and with other opcodes are used for recording:
%   frame_id - frame on which the event happened
%     get_trace - the total trace length in frames; if less than the actual max recorded 
%       frame number then will be extended to that value
%   stable_id - stable ID of the ROI that stayed through the operation, or for mod_shard
%     the ID of the new shard (since it has its own independent life afterwards)
%       get_trace - stable ID of the ROI to extract
%   brightness - brightness or its change of the ROI with stable_id, by opcode:
%     get_trace - the reference brightness at the end of recording, used for
%       adjustment of brightnesses of intermediate events
%     detect[_early] - the detected brightness, fitting the ROI into the frame
%       - norm_mode 0, 1: Brightness per normalized2
%       - norm_mode 2: Brigghtness per normalized3
%     mod_merge - norm_mode 0: fitting the merged ROI form into the old one
%       - norm_mode 1: total(first old ROI) / total(merged ROI)
%       - norm_mode 2: brightness(first old ROI) / brightness(merged ROI)
%     mod_subtract - norm_mode 0: fitting the new ROI form into the old one
%       - norm_mode 1: total(old ROI) / total(remaining ROI)
%       - norm_mode 2: brightness(old ROI) / brightness(remaining ROI)
%     mod_shard - norm_mode 0: fitting the new ROI form into the parent one
%       - norm_mode 1: total(old ROI) / total(shard ROI)
%       - norm_mode 2: brightness(old ROI) / brightness(shard ROI)
%   stable_id2 - stable ID of the other ROI that participated in the operation, by opcode:
%     get_trace - normalization mode
%     detect[_early] - 0
%     mod_merge - the ROI that got merged in and disappeared
%     mod_subtract - the ROI that got subtracted
%     mod_shard - the parent ROI
%   brightness2 - brightness or its change of the ROI with stable_id2, by opcode:
%     get_trace - unused
%     detect[_early] - norm_mode 0: normalization2 coefficient used by the ROI at this frame
%       - norm_mode 1: total(ROI) at this frame
%       - norm_mode 2: normalization3 (brightness) coefficient used by the ROI at this frame
%     mod_merge - norm_mode 0: fitting the merged ROI form into the second old one
%       - norm_mode 1: total(second old ROI) / total(merged ROI)
%       - norm_mode 2: brightness(second old ROI) / brightness(merged ROI)
%     mod_subtract - 0
%     mod_shard - 0
%
% Returns:
% Nothing for most opcodes.
% on "get" - a cell matrix, where each row contains elements of events:
%   1 - opcode
%   2 - frame_id
%   3 - stable_id
%   4 - brightness
%   5 - stable_id2
%   6 - brightness2
function res = rois_event(opcode, varargin)
	persistent events;
	persistent pos;
	persistent use_native;

	if opcode == "clear"
		events = [];
		pos = [];
		if length(varargin) > 0 && varargin{1} 
			use_native = 1;
			rois_event_native('clear');
		else
			if use_native
				rois_event_native('clear');
			end
			use_native = 0;
		end
		res = [];
		return
	end

	if isempty(use_native) || ~use_native
		% the classic implementation without Triceps

		% Appending to large arrays is very slow, so amortize it by appending a large
		% number of empty slots at once.
		if isempty(pos)
			pos = 0;
		end
		if pos >= length(events)
			events = [events; cell(10000, 6)];
		end

		res = [];
		if opcode == "clear"
			events = [];
			pos = [];
		elseif opcode == "get"
			res = events(1:pos, :);
		elseif opcode == "get_trace"
			res = get_trace(events, pos, varargin{1}, varargin{2}, varargin{3}, varargin{4});
		elseif opcode == "get_last_frame"
			if pos > 0
				res = events{pos, 2};
			else
				res = 0;
			end
		elseif opcode == "set"
			events = cell2struct(varargin(1), "nest", 1).nest;
			pos = size(events, 1);
			% avoid growth by may cells on the next call that is likely to be "get"
			events = [events; cell(1, 6)];
		else
			pos = pos + 1;
			% varargin is a cell matrix, so its part need to be un-celled to show nicely
			frame_id = varargin{1};
			stable_id = varargin{2};
			brightness = varargin{3};
			stable_id2 = varargin{4};
			brightness2 = varargin{5};
			events(pos, :) =  {char(opcode), frame_id, stable_id, brightness, stable_id2, brightness2};
		end
	else
		% The Triceps events
		res = rois_event_native(char(opcode), varargin{:});
	end
end

% Get the trace for one ROI
%
% Args:
% evs - all events
% numev - number of valid events in evs
% last_frame - the total trace length in frames; if less than the actual max recorded 
%       frame number then will be extended to that value
% stable_id - stable id of the ROI to extract the trace for
% coeff - the reference brightness of ROI at the end of recording, used for
%   adjustment of brightnesses of intermediate events; the meaning depends on normalization mode
% norm_mode - normalization mode, see in rois_params()
function trace = get_trace(evs, numev, last_frame, stable_id, coeff, norm_mode)
	debug = 0;

	if numev > 0
		last_event_frame = evs{numev, 2};
	else
		last_event_frame = 0;
	end

	if length(last_frame) == 0 || last_frame < last_event_frame
		last_frame = last_event_frame
	end

	trace = zeros(1, last_frame);

	if numev == 0
		return % nothing to do
	end

	% the stable ids list starts with the stable id of the final ROI
	sids = [stable_id];
	% adjs is only used in norm_mode 1, 2
	adjs = [coeff];

	% flags: which events are fully detected
	is_true = int16(zeros(1, last_frame));

	% go through the trace backwards and try to track the history

	for evi = numev:-1:1
		% stable id from the event
		evsid = evs{evi, 3};

		% find if any of the interesting stable ids were updated here
		for sidx = find(sids == evsid)
			opcode = evs{evi, 1};
			frame = evs{evi, 2};
			if debug
				disp("Event " + string(evi) + " frame " + string(frame) + " sid " + string(sids(sidx)) + " op " + string(opcode));
			end

			% In norm_mode 1: the total weight of the ROI at the current point,
			% against which happen the adjustments of detection
			% In norm_mode2: the brightness of the ROI at the current point,
			% against which happen the adjustments of detection
			cur_adj = adjs(sidx);

			if opcode == "detect" % || opcode == "detect_early"
				coef2 = evs{evi, 6};
				if norm_mode == 0
					brightness = evs{evi, 4} / coef2 * coeff;
				elseif norm_mode == 1
					% The higher the total weight now compared to the total weight
					% at the detection time (that is in coef2), the lower would be
					% the detected brightness.
					brightness = evs{evi, 4} * coef2 / cur_adj;
				elseif norm_mode == 2
					% already contains the absolute brightness, no adjustment
					brightness = evs{evi, 4};
				else
					error "Unsupported normalization mode " + string(norm_mode);
				end

				if debug
					disp("Process detect " + string(brightness));
				end
				% SBXXX if trace(1, frame) < brightness || ~is_true(frame)
				% in mode 2 the early detection has the same rank as normal one
				if trace(1, frame) < brightness || (norm_mode ~=2 && ~is_true(frame))
					trace(1, frame) = brightness;
					is_true(frame) = 1;
				end
			elseif opcode == "detect_early"
				if ~is_true(frame)
					coef2 = evs{evi, 6};
					if norm_mode == 0
						brightness = evs{evi, 4} / coef2 * coeff;
					elseif norm_mode == 1
						% The higher the total weight now compared to the total weight
						% at the detection time (that is in coef2), the lower would be
						% the detected brightness.
						brightness = evs{evi, 4} * coef2 / cur_adj;
					elseif norm_mode == 2
						% already contains the absolute brightness, no adjustment
						brightness = evs{evi, 4};
						% SBXXX or do the adjustment
						% brightness = evs{evi, 4} * coef2 / cur_adj;
					else
						error "Unsupported normalization mode " + string(norm_mode);
					end
					if debug
						disp("Process detect_early " + string(brightness));
					end
					if trace(1, frame) < brightness
						trace(1, frame) = brightness;
					end
				end
			elseif opcode == "mod_merge"
				othersid = evs{evi, 5};
				if debug
					disp("Process merge " + string(othersid));
				end
				if othersid >= 0
					sids = [sids, othersid];
					% adjs is only used in norm_mode 1, 2
					adjs = [adjs, cur_adj * evs{evi, 6}];
				end
				% adjs is only used in norm_mode 1, 2
				adjs(sidx) = cur_adj * evs{evi, 4};
			elseif opcode == "mod_subtract"
				if debug
					disp("Process subtract " + string(othersid));
				end
				% adjs is only used in norm_mode 1, 2
				adjs(sidx) = cur_adj * evs{evi, 4};
			elseif opcode == "mod_shard"
				othersid = evs{evi, 5};
				if debug
					disp("Process shard " + string(othersid));
				end
				if othersid >= 0
					sids = [sids, othersid];
					% adjs is only used in norm_mode 1, 2
					adjs = [adjs, cur_adj * evs{evi, 4}];
				end
			end
		end
	end
end
