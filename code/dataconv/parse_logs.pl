#!/usr/bin/perl

use Carp;

my $NFRAMES = 41756;

print("n_rois,FPS,FPS_dev,CPU,CPU_dev\n");

for my $f (@ARGV) {
	printf(STDERR "file %s\n", $f);

	open(FILE, "<", $f) or confess "$!";

	my $nrois = 0;
	my $total_fps = 0.;
	my $total_cpu = 0.;
	my @fps;
	my @cpu;
	my $dev_fps = 0.;
	my $cpu_frame = 0.;
	my $dev_cpu = 0.;

	my $line = "";

	while (<FILE>) {
		if (/Speed average (\d+(\.\d*)?) FPS, recent (\d+(\.\d*)?) FPS, CPU recent u=(\d+(\.\d*)?)\% s=(\d+(\.\d*)?)\% all=(\d+(\.\d*)?)\%/) {
			push(@fps, $3);
			push(@cpu, $9);
		} elsif (/^Processing frame \d+, found (\d+)/) {
			$nrois = $1;
		} elsif (/^Speed average (\d+(\.\d*)?) FPS, CPU average u=(\d+(\.\d*)?)\% s=(\d+(\.\d*)?)\% all=(\d+(\.\d*)?)\%/) {
			my $total_fps = $1;
			my $total_cpu = $7;
			printf(STDERR "  fps %f cpu %f nrois %d\n", $total_fps, $total_cpu, $nrois);

			# This is end of run, so compute the summaries.

			$cpu_frame = $total_cpu / $total_fps / 100.;

			for (my $i = 0; $i <= $#fps; $i++) {
				$dev_fps += ($fps[$i] - $total_fps)**2;
				$dev_cpu += ($cpu[$i] / $fps[$i] / 100. - $cpu_frame)**2;
			}

			$dev_fps = sqrt($dev_fps / $#fps); # this is N-1
			$dev_cpu = sqrt($dev_cpu / $#cpu); # this is N-1

			printf(STDERR "  fps %f (std %f) cpu %f (%f) nrois %d\n", $total_fps, $dev_fps, $cpu_frame, $dev_cpu, $nrois);
			$line = sprintf("%d,%f,%f,%f,%f\n", $nrois, $total_fps, $dev_fps, $cpu_frame, $dev_cpu);

			# reset the values in case if there is one more run
			$nrois = 0;
			$total_fps = 0.;
			$total_cpu = 0.;
			@fps;
			@cpu;
			$dev_fps = 0.;
			$cpu_frame = 0.;
			$dev_cpu = 0.;
		}
	}

	# print from the last run
	print($line);

	close(FILE);
}
