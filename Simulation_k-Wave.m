% Generate simulated PAT scan data using k-wave
% Scan with single element ultrasound transducer (SUT)
% Full circular scanning, clockwise, Point target numerical phantom
% k-Wave Version 1.2.1 and Matlab R2017a
clear all; close all; clc;
%% parameters
medium.sound_speed = 1500; % sound speed in water [m/s]
outputfilename = ‘PAT_scan_data.txt’;
radius = 15e-3; % SUT radius in [m]
num_A_lines = 100; % number of A-lines
sensor.frequency_response = [5e6 70]; % 2.25 MHz center frequency SUT
% 70% nominal bandwidth
fs = 20; % sampling frequency in [MHz] or [MS/s]
Nsample = 400; % number of sample in each A-line data acquisition
object_sim.Nx = 341; % number of grid points in the x (row) direction
object_sim.Ny = 341; % number of grid points in the y (column) direction
object_sim.x = 34.1e-3; % total grid size [m]
object_sim.y = 34.1e-3; % total grid size [m]
% UST_pos = [0, 0]; % First SUT position (coordinates) [m]
%%
time.dt = 1/(fs*1e6); % sampling time in sec
time.length = Nsample; % number of points in time
time.t_array = 0:1:time.length-1; % time array of Nsample time steps
time.t_array = time.t_array*time.dt;
Nx = object_sim.Nx;
Ny = object_sim.Ny;
dx = object_sim.x/object_sim.Nx; % grid point spacing in the x direction
dy = object_sim.y/object_sim.Ny; % grid point spacing in the y direction
kgrid = kWaveGrid(object_sim.Nx, dx, object_sim.Ny, dy);
kgrid.t_array = (0:time.length-1)*time.dt;
% cart_sensor_mask = makeCartCircle(radius, num_A_lines, SUT_pos, 2*pi,1);
cart_sensor_mask = makeCartCircle(radius, num_A_lines);
sensor.mask = cart_sensor_mask;
%% Creating numerical point targets and generating PAT scan data
% 5 point target source placed along the x axis
source.p0 = zeros(object_sim.Nx, object_sim.Ny);
source.p0(170, 170) = 1; source.p0(170, 230) = 1; source.p0(170, 280) = 1;
source.p0(170, 120) = 1; source.p0(170, 70) = 1;
% set the input options for k wave
input_args = {‘Smooth’, false, ‘PMLInside’, false, ‘PlotPML’, false};
% run the k-wave simulation
PAT_data = kspaceFirstOrder2D(kgrid,medium,source,sensor,input_args{:});
%% Display
% plot SUT positions and the target object
figure; imagesc(kgrid.y_vec * 1e3, kgrid.x_vec * 1e3, source.p0 + . . .
cart2grid(kgrid,cart_sensor_mask), [-1, 1]);
colormap(getColorMap);
set(gca,’LineWidth’,2,’XTick’,[-15 -5 5 15],’YTick’, [-15 -5 5 15],. . .
‘fontweight’,’bold’,’fontsize’,12);
title(‘SUT positions & point targets’,’fontweight’,’bold’,’fontsize’,14);
ylabel(‘x-position [mm]’,’fontweight’,’bold’,’fontsize’,12);
xlabel(‘y-position [mm]’,’fontweight’,’bold’,’fontsize’,12); axis image;
% plot A-line PA signal
figure, plot(PAT_data(50,:),’LineWidth’,2);
set(gca,’LineWidth’,2,’XTick’,[50 150 250 350],’YTick’, [-4e-3 0 4e-3],. . .
‘fontweight’,’bold’,’fontsize’,12);
title(‘A-line PA signal’,’fontweight’,’bold’,’fontsize’,14);
xlabel(‘Time points’,’fontweight’,’bold’, ‘fontsize’,12);
ylabel(‘A-line PA signal amplitude’,’fontweight’,’bold’, ‘fontsize’,12);
% plot PAT Sinogram
figure, imagesc(PAT_data);
set(gca,’LineWidth’,2,’XTick’,[50 150 250 350],’YTick’,[10 30 50
70 90],. . .
‘fontweight’,’bold’,’fontsize’,12);
title(‘PAT Sinogram’,’fontweight’,’bold’,’fontsize’,14);
xlabel(‘Time points’,’fontweight’,’bold’, ‘fontsize’,12);
ylabel(‘SUT positions’,’fontweight’,’bold’, ‘fontsize’,12);
save(outputfilename, ‘PAT_data’, ‘-ascii’);