% A Boundex-Boxed Matrix: a sparse matrix where all the non-0
% values are contained inside a bounding box
classdef bbm
	properties
		% size of the full matrix
		ysz = 0;
		xsz = 0;
		% position of the bounding box
		ymin = 0;
		xmin = 0;
		% size of the bounding box
		ybbsz = 0;
		xbbsz = 0;
		% data inside the bounding box, of size (ybbsz, xbbsz)
		data = [];
	end
	methods
		% construct from either a plain matrix or another bboxed one
		function obj = bbm(arg, varargin)
			cls = string(class(arg));
			if cls == "double" || cls == "single"
				obj.ysz = size(arg, 1);
				obj.xsz = size(arg, 2);

				xcount = sum(arg ~= 0, 1);
				obj.xmin = find(xcount, 1);
				if length(obj.xmin) == 0
					% an empty matrix
					obj.ymin = 0;
					obj.xmin = 0;
					obj.ybbsz = 0;
					obj.xbbsz = 0;
					obj.data = [];
				else
					obj.xbbsz = find(xcount, 1, 'last') + 1 - obj.xmin;

					ycount = sum(arg ~= 0, 2);
					obj.ymin = find(ycount, 1);
					obj.ybbsz = find(ycount, 1, 'last') + 1 - obj.ymin;

					obj.data = arg(obj.ymin:obj.ymin+obj.ybbsz-1, obj.xmin:obj.xmin+obj.xbbsz-1);
				end
			elseif cls == "bbm" || cls == "struct"
				obj = arg;
				% the version below can be used to enforce the right struct fields
				% obj.ysz = arg.ysz;
				% obj.xsz = arg.xsz;
				% obj.ymin = arg.ymin;
				% obj.xmin = arg.xmin;
				% obj.ybbsz = arg.ybbsz;
				% obj.xbbsz = arg.xbbsz;
				% obj.data = arg.data;
			else
				error ("Can't construct a bbm from class " + cls);
			end
		end

		% convert to a plain matrix
		function plain = toPlain(obj)
			if obj.xsz == obj.xbbsz && obj.ysz == obj.ybbsz
				plain = obj.data
			else
				plain = zeros(obj.ysz, obj.xsz);
				plain(obj.ymin:obj.ymin+obj.ybbsz-1, obj.xmin:obj.xmin+obj.xbbsz-1) = obj.data;
			end
		end

		% transpose the matrix
		function res = T(obj)
			res = bbm([]);
			res.ysz = obj.xsz;
			res.xsz = obj.ysz;
			res.ymin = obj.xmin;
			res.xmin = obj.ymin;
			res.ybbsz = obj.xbbsz;
			res.xbbsz = obj.ybbsz;
			res.data = obj.data';
		end

		% apply an operation on the bboxed part
		% (technically, anyone can just apply it in place on obj.data, but this
		% produces a copy bbm with result, leaving the original bbm unchanged)
		function res = apply(obj, op)
			res = bbm([]);
			res.ysz = obj.ysz;
			res.xsz = obj.xsz;
			res.ymin = obj.ymin;
			res.xmin = obj.xmin;
			res.ybbsz = obj.ybbsz;
			res.xbbsz = obj.xbbsz;
			res.data = op(obj.data);
		end

		% Change to a different bounding box, in preparation for a new
		% operation. This will cut or fill with 0s as needed.
		% The original object is left unchanged, a rebboxed copy is returned.
		function res = rebbox(obj, ymin, xmin, ymax, xmax)
			if xmax > obj.xsz || ymax > obj.ysz
				error(sprintf("Object rebbox to size (%d, %d) is larger than original size (%d, %d)", ...
					ymax, xmax, obj.ysz, obj.xsz));
			end

			res = bbm([]);
			res.ysz = obj.ysz;
			res.xsz = obj.xsz;

			if ymax < ymin || xmax < xmin
				% empty result
				return
			end

			res.ymin = ymin;
			res.xmin = xmin;
			res.ybbsz = ymax - ymin + 1;
			res.xbbsz = xmax - xmin + 1;

			if xmin >= obj.xmin && xmax < obj.xmin + obj.xbbsz && ymin >= obj.ymin && ymax < obj.ymin+obj.ybbsz
				% no expansion, a pure cutting
				res.data = obj.data(ymin-obj.ymin+1 : ymax-obj.ymin+1, xmin-obj.xmin+1 : xmax-obj.xmin+1);
			else
				% expand with 0s
				res.data = zeros(res.ybbsz, res.xbbsz);

				% intersecting portion
				yismin = max(ymin, obj.ymin);
				yismax = min(ymax, obj.ymin+obj.ybbsz-1);
				xismin = max(xmin, obj.xmin);
				xismax = min(xmax, obj.xmin+obj.xbbsz-1);

				if yismax >= yismin && xismax >= xismin
					res.data(yismin-ymin+1:yismax-ymin+1, xismin-xmin+1:xismax-xmin+1) =  ...
						obj.data(yismin-obj.ymin+1:yismax-obj.ymin+1, xismin-obj.xmin+1:xismax-obj.xmin+1);
				end
			end
		end

		% check whether two bounding boxes have an intersection
		function res = hasIntersect(obj, obj2)
			% intersecting portion
			yismin = max(obj2.ymin, obj.ymin);
			yismax = min(obj2.ymin+obj2.ybbsz-1, obj.ymin+obj.ybbsz-1);
			xismin = max(obj2.xmin, obj.xmin);
			xismax = min(obj2.xmin+obj2.xbbsz-1, obj.xmin+obj.xbbsz-1);

			res = yismax >= yismin && xismax >= xismin
		end

		% apply an operation on intersection of two objects,
		% op must be 2-argument, arguments matrices of identical size,
		% and return the result of the same size
		function res = intersect(obj, obj2, op)
			if obj.ysz ~= obj2.ysz || obj.xsz ~= obj2.xsz
				error(sprintf("Intersecting matrices of different size (%d, %d) and (%d, %d)", ...
					obj.ysz, obj.xsz, obj2.ysz, obj2.xsz));
			end

			% intersecting portion
			yismin = max(obj2.ymin, obj.ymin);
			yismax = min(obj2.ymin+obj2.ybbsz-1, obj.ymin+obj.ybbsz-1);
			xismin = max(obj2.xmin, obj.xmin);
			xismax = min(obj2.xmin+obj2.xbbsz-1, obj.xmin+obj.xbbsz-1);

			res = bbm([]);
			res.ysz = obj.ysz;
			res.xsz = obj.xsz;

			if yismax >= yismin && xismax >= xismin
				% non-empty result
				res.ymin = yismin;
				res.xmin = xismin;
				res.ybbsz = yismax-yismin+1;
				res.xbbsz = xismax-xismin+1;
				res.data = op( ...
					obj.data(yismin-obj.ymin+1 : yismax-obj.ymin+1, xismin-obj.xmin+1 : xismax-obj.xmin+1), ...
					obj2.data(yismin-obj2.ymin+1 : yismax-obj2.ymin+1, xismin-obj2.xmin+1 : xismax-obj2.xmin+1));
			end
		end

		% Cut obj to an intersection with obj2 (the originals are unchanged, the
		% cut copy of obj is returned).
		function res = toIntersect(obj, obj2)
			if obj.ysz ~= obj2.ysz || obj.xsz ~= obj2.xsz
				error(sprintf("Intersecting matrices of different size (%d, %d) and (%d, %d)", ...
					obj.ysz, obj.xsz, obj2.ysz, obj2.xsz));
			end

			% intersecting portion
			yismin = max(obj2.ymin, obj.ymin);
			yismax = min(obj2.ymin+obj2.ybbsz-1, obj.ymin+obj.ybbsz-1);
			xismin = max(obj2.xmin, obj.xmin);
			xismax = min(obj2.xmin+obj2.xbbsz-1, obj.xmin+obj.xbbsz-1);

			res = bbm([]);
			res.ysz = obj.ysz;
			res.xsz = obj.xsz;

			if yismax >= yismin && xismax >= xismin
				% non-empty result
				res.ymin = yismin;
				res.xmin = xismin;
				res.ybbsz = yismax-yismin+1;
				res.xbbsz = xismax-xismin+1;
				res.data = obj.data(yismin-obj.ymin+1 : yismax-obj.ymin+1, xismin-obj.xmin+1 : xismax-obj.xmin+1);
			end
		end

		% Expand obj to a union with obj2 (the originals are unchanged, the
		% expanded copy of obj is returned).
		function res = toUnion(obj, obj2)
			if obj.ysz ~= obj2.ysz || obj.xsz ~= obj2.xsz
				error(sprintf("Unioning matrices of different size (%d, %d) and (%d, %d)", ...
					obj.ysz, obj.xsz, obj2.ysz, obj2.xsz));
			end
			
			if obj2.xmin == 0
				% unioning with an empty bbox makes no change
				res = obj;
				return;
			end

			if obj.xmin == 0
				% copy dimensions from the non-empty object
				yismin = obj2.ymin;
				yismax = obj2.ymin+obj2.ybbsz-1;
				xismin = obj2.xmin;
				xismax = obj2.xmin+obj2.xbbsz-1;
			else
				% unioning portion
				yismin = min(obj2.ymin, obj.ymin);
				yismax = max(obj2.ymin+obj2.ybbsz-1, obj.ymin+obj.ybbsz-1);
				xismin = min(obj2.xmin, obj.xmin);
				xismax = max(obj2.xmin+obj2.xbbsz-1, obj.xmin+obj.xbbsz-1);
			end

			res = obj.rebbox(yismin, xismin, yismax, xismax);
		end

		% multiply two bboxed matrices
		function res = mul(obj, obj2)
			if obj.xsz ~= obj2.ysz 
				error(sprintf("Multiplying matrices (%d, %d) and (%d, %d) of mismatching dimensions %d~=%d", ...
					obj.ysz, obj.xsz, obj2.ysz, obj2.xsz, obj.xsz, obj2.ysz));
			end

			res = bbm([]);
			res.ysz = obj.ysz;
			res.xsz = obj2.xsz;

			% intersection on the direction of summation
			ismin = max(obj.xmin, obj2.ymin);
			ismax = min(obj.xmin+obj.xbbsz-1, obj2.ymin+obj2.ybbsz-1);

			if ismax >= ismin && obj.ybbsz > 0 && obj2.xbbsz > 0
				% non-empty result
				res.ymin = obj.ymin;
				res.xmin = obj2.xmin;
				res.ybbsz = obj.ybbsz;
				res.xbbsz = obj2.xbbsz;
				res.data =  obj.data(:, ismin-obj.xmin+1 : ismax-obj.xmin+1) ...
					* obj2.data(ismin-obj2.ymin+1 : ismax-obj2.ymin+1, :);
			end
		end

	end
end
