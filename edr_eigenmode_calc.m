% currently only for left hemisphere and cortex

addpath(genpath('matlab_functions'));

% surface
surface_interest = 'fsLR_32k';
hemisphere = 'lh';
mesh_interest = 'midthickness';

[vertices, faces] = read_vtk(sprintf('data/template_surfaces_volumes/%s_%s-%s.vtk', surface_interest, mesh_interest, hemisphere));
surface_to_analyze.vertices = vertices';
surface_to_analyze.faces = faces';

% cortex mask
cortex = dlmread(sprintf('data/template_surfaces_volumes/%s_cortex-%s_mask.txt', surface_interest, hemisphere));
cortex_ind = find(cortex);

num_vertices = length(cortex);
num_modes = 200;

connectome = load(sprintf('data/empirical/S255_high-resolution_group_average_connectome_cortex_nomedial-%s.mat', hemisphere));
connectome = connectome.avgSC_L; % left hemisphere

disp('files loaded.');


%% EDR EIGENMODES
disp('BEGINNING EDR EIGENMODES CALCULATION');

% connection lengths of vertices without medial wall
euclidean_dist = squareform(pdist(surface_to_analyze.vertices(cortex_ind,:)));

% estimate true length
sigmoid = @(p, x) p(1) * x + p(2) + p(3) ./ (1 + exp(-p(4) * (x - p(5))));
p = [1, 5, 45, 0.15, 35];
surface_dist = sigmoid(p, euclidean_dist);

disp('Connection lengths calculated.');

% probability function for connection
Pspace_func = @(scale, distance) exp(-scale*distance);

% generate pseudorandom numbers to compare with probability function
rng(1) % set for reproducibility
rand_prob = rand(size(surface_dist));
rand_prob = triu(rand_prob,1) + triu(rand_prob,1)';
rand_prob(1:1+size(rand_prob,1):end) = 1;

% empirical scale exponent parameter based on structural connectivity
connectome_weights = connectome(:);
connectome_distances = surface_dist(:);

exp_function = @(alpha, d) exp(-alpha * d);
error_function = @(alpha) sum((connectome_weights - exp_function(alpha, connectome_distances)).^2);

alpha_empirical = 0.1668;

scale = alpha_empirical;
Pspace = Pspace_func(scale, surface_dist);
Pspace(1:1+size(Pspace,1):end) = 0;
Pspace = Pspace/max(Pspace(:));

% generate EDR connectome
connectome = double(rand_prob < Pspace);
connectome(1:1+size(connectome,1):end) = 0;

disp('Synthetic EDR connectome generated.');

N = size(connectome,1);
connection_density = nnz(connectome) / (N * (N-1)) * 100; % percentage
disp(connection_density);

% =========================================================================
%                           Calculate the modes                            
% =========================================================================

% speed things up
clearvars -except connectome cortex_ind connection_density

% [eig_vec_temp, eig_val] = calc_network_eigenmode(connectome, 200);
d = sum(connectome,2);
D = diag(d);
L = D - connectome;
clear D;
clear connectome;

Dhalf = diag(1./sqrt(d));
clear d;
Lnorm = Dhalf*(L*Dhalf);
clear L;
clear Dhalf

Lnorm(isnan(Lnorm)) = 0;

[eig_vec_temp, eig_val] = eig(Lnorm);
clear Lnorm;

eig_vec_temp = eig_vec_temp(:,1:200);
eig_val = diag(eig_val(1:200, 1:200));

% return medial wall vertices
eig_vec = zeros(32492, 200);
eig_vec(cortex_ind,:) = eig_vec_temp(:,1:200);
save(sprintf('data/eigenmodes/EDR_eigenmodes-%s_%i.mat', hemisphere, alpha_empirical*10000), 'eig_val', 'eig_vec', '-v7.3')

disp('EDR modes calculated');