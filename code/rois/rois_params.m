function params = rois_params(varargin)
	p = inputParser;

	% --- Events from a previous partial run.

	p.addParameter('events', {});

	% --- Controls for debugging/optimization

	% Enable the trace of all the merging between ROIs. Useful for investigation
	% but can use A LOT of memory.
	p.addParameter('merge_trace', 0);

	% Write the diary into the log file.
	p.addParameter('logfile', []);

	% Use Triceps native implementation for event collection.
	p.addParameter('event_triceps', 1);

	% --- Temporal parameters.

	% This many frames (starting with the "current" one) get averaged
	% before processing in SEUDO.
	p.addParameter('avg_frames', 5);

	% This many frames (after averaging) are considered "recent", and the
	% temporally-near logic is used to combine the candidate ROIs found
	% in them.
	p.addParameter('recent_frames', 10);

	% --- Controls of ROI detection

	% Keep an extended state to try combining better. EXPERIMENTAL
	p.addParameter('extended_state', 0);

	% The options 'online' and 'offline' together determine the work mode:
	%  neither is set - build the list of ROIs but don't perform detection
	%    of the activations.
	%  online is set - build the list of ROIs and detect their activation
	%    in rois_event() along the way (the activation traces will be
	%    somewhat dirty as the ROIs are getting adjusted when more information
	%    gets discovered).
	%  offline is set - don't build the list of ROIs, use the one given in
	%    the arguments, and do the detection with it. Make sure to set the
	%    other parameters the same across the building the list of ROIs and
	%    detection with it.
	%  both are set - invalid.
	p.addParameter('online', 0);
	p.addParameter('offline', 0);

	% Flush the near ROIs at the end, combining them all into the far ones.
	% This makes sure that the ROIs that stay lighted all along (which can
	% happen in the simulations) get reported too.
	p.addParameter('flush', 0);

	% If > 1, will do the adjustment for local variations of brightness
	% by computing the median in chunks, with the chunk size computed by
	% dividing the image size in each dimension by split ratio for that
	% dimension. Setting only one ratio to > 1 automatically causes the
	% other ratio to be set to the same value. Since the chunks are tiled
	% with overlapping by half, the actual number of used chunks will be twice
	% higher in each dimension.
	p.addParameter('med_chunks_x', -1);
	p.addParameter('med_chunks_y', -1);

	% Minimum number of pixels in a ROI (including those included by mask blurring).
	p.addParameter('min_roi_size', 50);

	% Minimum brightness of "proper" pixels in a ROI (those included before mask blurring);
	% values >=0 set an absolute limit, values < 0 set the limit as multiples
	% of noise amplitude (level above the median in a frame by as much as median
	% is above bottom)
	p.addParameter('min_proper_px', -1);

	% Minimum average brightness of pixels in a ROI;
	% values >=0 set an absolute limit, values < 0 set the limit as multiples
	% of noise amplitude (level above the median in a frame by as much as median
	% is above bottom)
	p.addParameter('min_avg_px', -1);

	% Blur radius in pixels for deteriming the mask of a ROI, using the radius of 0
	% makes all the pixels in a ROI to at least touch corners, the blurring covers
	% the disjointedness and includes pixels around boundaries.
	p.addParameter('mask_blur_rad', 1);

	% Flag: do the proper seudo with blobs (Gaussian kernels). Setting to 0 will
	% do a simplified (but faster) version without blobs, with only the ROIs being fit.
	p.addParameter('seudo', 1);

	% Flag: before detecting ROIs in a frame, blobify it by representing the image as
	% a fitted set of blobs, then add together these blobs, thus smoothing the image.
	% -1 means to copy the value from seudo.
	p.addParameter('blobify', -1);

	% Gaussian radius (the actual radius would be higher) of the blobs used for
	% SEUDO and blobification.
	p.addParameter('blobify_rad', 1.2);

	% --- Patchwork mode, where the movie represents only a spatial patch of full movie.

	% After collecting back the patches, merge the ROIs at the patch boundaries.
	p.addParameter('patch_merge', 1);

	% If 1, the patch processing runs in parallel as normal, otherwise runs sequentially
	% for debugging.
	p.addParameter('patch_parallel', 1);

	% If > 0, specifies the number of processes for patches, the default
	% is equal to the number of patches.
	p.addParameter('patch_nproc', -1);

	%-- The rest of patch parameters are set internally by patch_find_movie_rois()
	% and not normally set manually.

	% If all parameters >= 0 (and cannot be one < 0 and another >= 0), enables the patch
	% mode and sets the offset of the movie patch in the full frame. The size of
	% the patch is taken from the size of the movie. The ROIs are generated with
	% dimensions for the full movie. The input ROIs are filtered by dimensions,
	% and those that do not fully fit into the patch are discarded.

	% Offsets of the patch in the movie.
	% The offsets are 0-based (unlike Matlab indexes).
	%
	% Note that the movie dimensions go (Y, X, N), which is important when cutting
	% the patch.
	p.addParameter('patch_offset_x', -1);
	p.addParameter('patch_offset_y', -1);

	% The size of the greater movie. The returned ROIs will be sized to it.
	p.addParameter('movie_wd', -1);
	p.addParameter('movie_ht', -1);

	% the id of the patch
	p.addParameter('patch_id', -1);

	% --- Controls for ROI combining and subtraction.

	% If the temporally-near old and new ROI have at least this fraction of the area
	% in common, they can be combined.
	p.addParameter('combine_near_min_common', 0.75);

	% If the old and new ROI have this much in common, unify them,
	% the same criteria is used for subtraction but when the relation
	% between ROIs is asymmetric, one side being above and the other below.
	% "This much" is not a fraction of area here but a relation between
	% the fit scores, the fit of one ROI to another as-is is the numerator
	% while the fit of one ROI to another in only the common area as
	% denominator. For a merge, both ratios need to be above the limit,
	% for subtraction only one ratio.
	p.addParameter('combine_far_min_common', 0.75);
	% When subtracting, the ratio of brightness of larger ROI to smaller one
	% should not be above thisi value. This prevents the pale ROIs from
	% being subtracted from the bright ones.
	p.addParameter('subtract_far_max_brightness_ratio', 2);
	% When subtracting, maximum value representing the symmetry of fitting
	% one side to the other (i.e. to allow subtraction the fitting has to
	% have a high enough asymmetry). The symmetry is expressed as ratio
	% of fit score of one ROI into another to the fit score of the second
	% ROI into first.
	p.addParameter('subtract_far_max_symmetry', 0.75);

	% --- Controls for FISTA fit by seudo_native().

	% During computation, place blobs on a grid of this many pixels
	% increase between the points, both vertically and horizontally. The weights for
	% blobs in between will be filled with 0s. If negative, the absolute value is used
	% to compute the pixel count as a fraction of smallest blob dimension. If positive,
	% is the pixel count and must be at least 1 pixel.
	% -0.2 is a moderately aggressive value for reasonably fast detection.
	% -0.3 is an aggressive value for more fast detection.
	p.addParameter('blob_spacing', -0.3);

	% LASSO lambda value for ROIs
	p.addParameter('fit_lambda_rois', 0);
	% LASSO lambda value for blobs
	p.addParameter('fit_lambda_blobs', 0.15);

	% Tolerance for the FISTA fit.
	p.addParameter('fit_tol', 0.01);

	% Maximal number of steps for FISTA.
	p.addParameter('fit_max_steps', 500);

	% Mode for L computation:
	% 0 for dynamic L (as in TFOCS), 1 for static multi-L,
	% 2 for dynamic L + fast brake, 3 for static multi-L + fast brake
	p.addParameter('fit_l_mode', 2);

	% Mode for stopping:
	% 0 for relative norm2 (as in TFOCS), 1 for norm2, 2 for every dimension
	p.addParameter('fit_stop_mode', 0);

	% Number of parallel threads to use in seudo_native().
	p.addParameter('fit_parallel', 16);

	% Report a detection event if a ROI was fit into the frame with at least this weight.
	% Think of it as if you're making a graph of ROI traces, this should be less than
	% one pixel on that graph. It reduces the number of events by eliminating the very
	% tiny activations. To see everything, set this parameter to 0.
	p.addParameter('min_detect_fit', 0.001);

	% Start reporting "detect_early" events if a tentative ROI was fit into the frame (after
	% subtraction of well-known ROIs) with at least this weight. After the reporting is
	% started, the following frames use the min_detect_fit limit.
	p.addParameter('min_detect_early_fit', 0.5);

	% Report "mod_merge" on the tentative ROIs only if the fit changes by more than this.
	p.addParameter('min_detect_early_mod', 0.001);

	% Normalization mode:
	%  0 - fit using normalized2, in online mode correct brightness on merging by
	%      the ratio of normalized2_coeff
	%  1 - fit using normalized2, in online mode correct brightness on merging by
	%      the ratio of total weight of ROIs
	%  2 - fit using normalized2, immediate correct to normalized3 (i.e. brightness as
	%      normalization coefficient), in online mode no correction on merging
	p.addParameter('norm_mode', 1);

	% --- Parse and postprocess.
	parse(p, varargin{:});
	params = p.Results;

	if params.online && params.offline
		error("The online and offline modes cannot be enabled together");
	end

	if params.norm_mode < 0 || params.norm_mode > 2 || params.norm_mode ~= floor(params.norm_mode)
		error("Incorrect norm_mode " + string(params.norm_mode) + ", must be in [0, 2]");
	end

	npatch = (params.patch_offset_x >= 0) + (params.patch_offset_y >= 0) + (params.movie_wd >= 0) + (params.movie_ht >= 0);
	if npatch ~= 0 && npatch ~= 4
		error("The patch and movie dimensions must be specified all or none");
	end

	if params.blobify < 0
		params.blobify = params.seudo;
	end

	% copied from estimateTimeCoursesWithSEUDO()
	cropRad     = ceil(params.blobify_rad*2.5+eps);
	clipHeight  = 0.01;
	one_blob = fspecial('gauss',(cropRad*2+1)*[1 1],params.blobify_rad);
	one_blob = one_blob .* double(one_blob > clipHeight * max(one_blob(:)));
	one_blob = one_blob/sqrt(sum(one_blob(:).^2));

	% Blob used for blobification of images.
	params.blobify_blob = one_blob;

	% Propagate one chunk dimension to another one.
	if params.med_chunks_x > 1 && params.med_chunks_y <= 1
		params.med_chunks_y = params.med_chunks_x;
	elseif params.med_chunks_y > 1 && params.med_chunks_x <= 1
		params.med_chunks_x = params.med_chunks_y;
	end
end
