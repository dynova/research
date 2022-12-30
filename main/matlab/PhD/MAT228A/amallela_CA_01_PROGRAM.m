%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Computing Assignment #1                                                 %
% Written by Abhishek Mallela                                             %
% Latest update: 10/25/18                                                 %
% Written in MATLAB 2018a - academic use license                          %
%                                                                         %
% This MATLAB script is complete and self-contained.                      %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Preliminaries
a = 3; % wave speed
t_final = 9.0; % final simulation time
sigma = 0.9; % CFL number
e_vec = 6:12; % vector of exponents 'k' in 2^k
len = length(e_vec);
M_vec = 2.^e_vec; % vector of 'M' values
h_vec = 1 ./ M_vec;
delta_t_vec = sigma .* h_vec ./ a;
num_timesteps_vec = t_final ./ delta_t_vec;
num_tests = 3; % Number of test cases
num_methods = 3; % Number of finite difference methods
num_norms = 3; % Number of norm cases

%% Preallocations with cell arrays
% Computed grid, exact data, computed data, intermediate data
[c_grid, data_e, data_c, data_i] = deal(cell(1, len));

% Cell array of function handles for test cases
ff = {@Psi_1, @Psi_2, @Psi_3};

% The idea in the following set of 'for' loops is this: I am creating a
% cell array of cell arrays. Each outer cell array consists of 7 elements,
% and each element is a (3 x 3) cell array consisting of vectors. The rows
% correspond to the test cases: Square, Semicircle, and Gaussian Pulse.
% Traversing the columns, we have the methods: Upwind, Lax-Friedrichs,
% and Lax-Wendroff. Each vector consists of the exact, computed, and
% intermediate solution values over the computed grid, respectively.
% The length of each vector is 2^k, for k ranging over 6 to 12. It turns
% out that in my implementation, it is beneficial to have "grand data"
% structures. They are not prohibitively large.

for ii = 1:len
    [data_e{ii}, data_c{ii}, data_i{ii}] = deal(cell(num_tests, num_methods));
    for jj = 1:num_tests*num_methods
        [data_e{ii}{jj}, data_c{ii}{jj}, data_i{ii}{jj}] = deal(zeros(1, M_vec(ii)));
    end
end

%% Finite difference schemes
for ind = 1:len
    M = M_vec(ind);
    h = h_vec(ind);
    c_grid{ind} = h*(1:M)-h/2;
    num_timesteps = num_timesteps_vec(ind);
    ii = 2:M-1;
    for kk = 1:num_tests
        for jj = 1:num_methods
            % Using the function handle for Psi_i; i = 1, 2, 3
            data_e{ind}{kk,jj} = ff{kk}(c_grid{ind});
            data_c{ind}{kk,jj} = ff{kk}(c_grid{ind});
            data_i{ind}{kk,jj} = ff{kk}(c_grid{ind});

        end
        for t = 1:num_timesteps
            data_c{ind}{kk,1}(ii) = data_i{ind}{kk,1}(ii) - sigma*(data_i{ind}{kk,1}(ii) - data_i{ind}{kk,1}(ii-1));
            data_c{ind}{kk,2}(ii) = (data_i{ind}{kk,2}(ii+1)+data_i{ind}{kk,2}(ii-1))/2 - (sigma/2)*(data_i{ind}{kk,2}(ii+1) - data_i{ind}{kk,2}(ii-1));
            data_c{ind}{kk,3}(ii) = (sigma/2)*(1+sigma)*data_i{ind}{kk,3}(ii-1)+(1-sigma^2)*data_i{ind}{kk,3}(ii)-(sigma/2)*(1-sigma)*data_i{ind}{kk,3}(ii+1);
            data_c{ind}{kk,1}(1) = data_i{ind}{kk,1}(1) - sigma*(data_i{ind}{kk,1}(1) - data_i{ind}{kk,1}(M));
            data_c{ind}{kk,2}(1) = (data_i{ind}{kk,2}(2)+data_i{ind}{kk,2}(M))/2 - (sigma/2)*(data_i{ind}{kk,2}(2) - data_i{ind}{kk,2}(M));
            data_c{ind}{kk,3}(1) = (sigma/2)*(1+sigma)*data_i{ind}{kk,3}(M)+(1-sigma^2)*data_i{ind}{kk,3}(1)-(sigma/2)*(1-sigma)*data_i{ind}{kk,3}(2);
            data_c{ind}{kk,1}(M) = data_i{ind}{kk,1}(M) - sigma*(data_i{ind}{kk,1}(M) - data_i{ind}{kk,1}(M-1));
            data_c{ind}{kk,2}(M) = (data_i{ind}{kk,2}(1)+data_i{ind}{kk,2}(M-1))/2 - (sigma/2)*(data_i{ind}{kk,2}(1) - data_i{ind}{kk,2}(M-1));
            data_c{ind}{kk,3}(M) = (sigma/2)*(1+sigma)*data_i{ind}{kk,3}(M-1)+(1-sigma^2)*data_i{ind}{kk,3}(M)-(sigma/2)*(1-sigma)*data_i{ind}{kk,3}(1);
            data_i = data_c;
        end
    end
end

%% Error computations
format long
[L1, L2, Linf] = deal(zeros(len,num_tests,num_methods));

for ii = 1:len
    for jj = 1:num_tests
        for kk = 1:num_methods
            L1(ii,jj,kk) = Compute_One_Norm(data_e{ii}{jj,kk},data_c{ii}{jj,kk},M_vec(ii));
            L2(ii,jj,kk) = Compute_Two_Norm(data_e{ii}{jj,kk},data_c{ii}{jj,kk},M_vec(ii));
            Linf(ii,jj,kk) = Compute_Max_Norm(data_e{ii}{jj,kk},data_c{ii}{jj,kk});
        end
    end
end

%% Rate computations
N = {'L1','L2','Linf'};
rates = zeros(len-1,num_tests,num_methods,num_norms);
for hh = 1:len-1
    for ii = 1:num_tests
        for jj = 1:num_methods
            for kk = 1:num_norms
                rates(hh,ii,jj,kk) = log2(eval([eval('N{kk}') '(hh,ii,jj)'])/eval([eval('N{kk}') '(hh+1,ii,jj)']));
            end
        end
    end
end

%% Table generation
MM = cell(num_norms,num_tests);
fid = zeros(num_norms,num_tests);
T = {'SQUARE_WAVE_', 'SEMICIRCLE_', 'GAUSSIAN_'};

for ii = 1:num_norms
    for jj = 1:num_tests
        MM{ii,jj} = zeros(len,5);
        MM{ii,jj}(:,1) = M_vec;
        MM{ii,jj}(:,2) = h_vec;
        MM{ii,jj}(:,3) = eval([eval('N{ii}') '(:,jj,1)']);
        MM{ii,jj}(:,4) = eval([eval('N{ii}') '(:,jj,2)']);
        MM{ii,jj}(:,5) = eval([eval('N{ii}') '(:,jj,3)']);
        fid(ii,jj) = fopen(['AMALLELA_' T{jj} N{ii} '_ERROR.csv'],'w');
        fprintf(fid(ii,jj),'%.6g,   %.6g,    %.6f, %.6f, %.6f \n',MM{ii,jj}(1,:));
        fprintf(fid(ii,jj),'%.6g,  %.6g,   %.6f, %.6f, %.6f \n',MM{ii,jj}(2,:));
        fprintf(fid(ii,jj),'%.6g,  %.6g,  %.6f, %.6f, %.6f \n',MM{ii,jj}(3,:));
        fprintf(fid(ii,jj),'%.6g,  %.7g, %.6f, %.6f, %.6f \n',MM{ii,jj}(4,:));
        fprintf(fid(ii,jj),'%.6g, %.6g, %.6f, %.6f, %.6f \n',MM{ii,jj}(5,:));
        fprintf(fid(ii,jj),'%.6g, %.6g, %.6f, %.6f, %.6f \n',MM{ii,jj}(6,:));
        fprintf(fid(ii,jj),'%.6g, %.6g, %.6f, %.6f, %.6f \n',MM{ii,jj}(7,:));
        fclose(fid(ii,jj));
    end
end

%% Generation of plot data
X = {'UPWIND_', 'LF_', 'LW_'};
R = '4096';
fid = zeros(num_methods,num_tests);
for ii = 1:num_methods
    for kk = 1:num_tests
        fid(ii,kk) = fopen(['amallela_' T{kk} X{ii} R '_STATE.txt'],'wt');
        for ll = 1:str2double(R)
            fprintf(fid(ii,kk),'%.6f  %.6f \n',[c_grid{end}(ll) data_c{end}{kk,ii}(ll)]);
        end
        fclose(fid(ii,kk));
    end
end

%% Output of plots for LaTeX report
init_conds = {'Square Wave:', 'Semicircle:', 'Gaussian Pulse:'};
methods = {'Upwind', 'LF', 'LW'};
colors = {'k','b','g','r'};
num_resolutions = 4;
counter = 1;
for jj = 1:num_methods
    for ii = 1:num_tests
        figure(counter)
        clf;
        counter = counter + 1;
        for kk = 1:num_resolutions
            hold on
            plot(c_grid{2*kk-1},data_e{2*kk-1}{ii,jj},colors{kk},'linewidth',2)
            plot(c_grid{2*kk-1},data_c{2*kk-1}{ii,jj},strcat(colors{kk},'+'),'MarkerSize',4)
            title(strcat(init_conds{ii},{' '},methods{jj}),'fontsize',16);
            legend('Exact: M = 2^6', 'Computed: M = 2^6', 'Exact: M = 2^8', 'Computed: M = 2^8',...
                'Exact: M = 2^{10}', 'Computed: M = 2^{10}', 'Exact: M = 2^{12}', 'Computed: M = 2^{12}',...
                'Location','Best');
            hold off
        end
        saveas(gcf,strcat('Figure', num2str(counter-1),'.pdf'));
    end
end

%% Subroutines
% Defines the initial data for the square wave.
function [psi_1] = Psi_1(z)
psi_1 = zeros(1,length(z));

for k = 1:length(z)
    if z(k) >= 0.25 && z(k) <= 0.75
        psi_1(k) = 1;
    else
        psi_1(k) = 0;
    end
end
end

% Defines the initial data for the semicircle.
function [psi_2] = Psi_2(z)
psi_2 = zeros(1,length(z));

for k = 1:length(z)
    psi_2(k) = (0.25-(z(k)-(0.5))^2)^((0.5));
end
end

% Defines the initial data for the Gaussian Pulse.
function [psi_3] = Psi_3(z)
psi_3 = zeros(1,length(z));

for k = 1:length(z)
    psi_3(k) = exp(-256*(z(k)-(0.5))^2);
end
end

% Computes the discrete one-norm of the difference u - v
% of the two grid functions u and v, each of which has M elements, and
% returns the result as a variable named one_norm.
function [one_norm] = Compute_One_Norm(u, v, M)
h = 1/M;
one_norm = 0;

for k = 1:M
    one_norm = one_norm + abs(u(k) - v(k));
end

one_norm = h * one_norm; % Multiply by 'h'
end

% Computes the discrete two-norm of the difference u - v
% of the two grid functions u and v, each of which has M elements, and
% returns the result as a variable named two_norm.
function [two_norm] = Compute_Two_Norm(u, v, M)
h = 1/M;
two_norm = 0;

for k = 1:M
    two_norm = two_norm + (abs(u(k) - v(k)))^2;
end

two_norm = sqrt(h * two_norm); % Scale by sqrt(h)
end

% Computes the discrete max-norm of the difference u - v
% of the two grid functions u and v and returns the result as a variable 
% named max_norm.
function [max_norm] = Compute_Max_Norm(u, v)
max_norm = max(abs(u - v));
end