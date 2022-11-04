
% tensor dictionary learning for SSFs representation
%
clc
clear all
close all
%% basic parameter settings
randn('seed', 1); rand('seed', 1);
% single source to single receiver
addpath(genpath(pwd))


%% Training 
disp('------------------TRAIN-------------------------------------')
%% TDL
%         :param M_n:columns of n-mode dictionary£¨M_n>I_n£©
%         :param U_Tucker{n}: n-mode dictionary
%         :param max_iter
%         :param tol: tolerance
%         :param n_nonzero_coefs

%% real world data
Ocean_depth = 3000;
load ssp_initial.mat; 
C_int=ssp_initial;
N_x = 20;N_y =20;N_z= 10; Day= 50;
C_mean_tensor = zeros(Day,N_x,N_y,N_z);
for k=1:size(C_int,1)
    C_int_k = squeeze(C_int(k,:,:,:));
    C_int_k_3 = double(tenmat(C_int_k,3));
    ssp_mean_3 = mean(C_int_k_3,2);
    for i = 1:size(C_int,2)
        for j=1:size(C_int,3)
            C_int(k,i,j,:) = squeeze(C_int(k,i,j,:))- ssp_mean_3;
            C_mean_tensor(k,i,j,:) = ssp_mean_3';
        end
    end
end


train_day = 1;
M_1=60;
M_2=60;
M_3=30;
I_1=20;
I_2=20;
I_3=10;
I_total=I_1*I_2*I_3;
ssp_train = squeeze(C_int(train_day,1:I_1,1:I_2,1:I_3));

% ________________________________________________________________________________
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
%initialize dictionary factor matrix
tolerence=1e-6;      %tomp threshold 
n_nonzero_coefs=8; % T

%% 1.initilize the dictionary
U0_TDL=cell(3,1);
 T_1=tenmat(ssp_train,1);
temp_index1=randperm(I_2*I_3);
U0_TDL{1}=T_1(:,temp_index1(1:M_1));
 T_2=tenmat(ssp_train,2);
 temp_index2=randperm(I_1*I_3);
U0_TDL{2}=T_2(:,temp_index2(1:M_2));
 T_3=tenmat(ssp_train,3);
  temp_index3=randperm(I_1*I_2);
U0_TDL{3}=T_3(:,temp_index3(1:M_3));



%% training phase
%TDL
 U_TDL=U0_TDL;
  max_iter=12;
 for i =1:max_iter
     fprintf('NO. %d. \n',i);
     if i==1
     [S,~] = TOMP_func2(ssp_train,U_TDL,tolerence,n_nonzero_coefs);
     end
     ssp_recon=lmlragen(U_TDL,S);               
     ssp_error=reshape(ssp_train,[],1)-reshape(ssp_recon,[],1);
     
     error=norm(ssp_error(:),2)./sqrt(I_total); 
     fprintf('error= %f. \n',error);

     % update dictionary
     [U,S_new]= TALS2_func(ssp_train,S,U_TDL,n_nonzero_coefs);
     U_TDL=U;
     S=S_new;
 end

ssp_recon_final=lmlragen(U_TDL,S);               %Reconstruct tensor
ssp_err_final = ssp_train-ssp_recon_final;
RMSE_final = norm(ssp_err_final(:),2)./sqrt(I_total);
fprintf('The training RMSE of  SSF representation is %f. \n', RMSE_final);

%HOOI
R_tucker_1 = 2; R_tucker_2 = 2; R_tucker_3 = 2;
core_size = [R_tucker_1, R_tucker_2,R_tucker_3];
U0_HOOI = cell(3,1);
[U01, ~,~]= svd(randn(I_1, core_size(1)));
U0_HOOI{1} = U01(:,1:core_size(1));
[U02, ~,~]= svd(randn(I_2, core_size(2)));
U0_HOOI{2} = U02(:,1:core_size(2));
[U03, ~,~]= svd(randn(I_3, core_size(3)));
U0_HOOI{3} = U03(:,1:core_size(3));
[U_HOOI_est, S_HOOI_est, output_tucker] = lmlra_hooi(ssp_train, U0_HOOI); 
    

%% test phase

HOOI_test_err=zeros(30,1);
TDL_test_err=zeros(30,1);
for test_day=1:30
    
ssp_test = squeeze(C_int( test_day,1:I_1,1:I_2,1:I_3));    
% HOOI test
mode = [1,2,3];
S_HOOI = tmprod(ssp_test,U_HOOI_est,mode,'H');
HOOI_recon_test = tmprod(S_HOOI,U_HOOI_est,mode);
HOOI_recon_err=reshape( ssp_test,[],1)- reshape(HOOI_recon_test,[],1);     
HOOI_test_err(test_day)   = norm(HOOI_recon_err(:),2)./sqrt(I_total);
 fprintf('%HOOI test error= %f. in %f-th day \n',HOOI_test_err(test_day),test_day);
 
% TDL test

[S_TDL,~] = TOMP_func2(ssp_test,U_TDL,tolerence,n_nonzero_coefs);
TDL_recon=lmlragen(U_TDL,S_TDL);               
TDL_error=reshape(ssp_test,[],1)-reshape(TDL_recon,[],1);    
TDL_test_err(test_day)=norm(TDL_error(:),2)./sqrt(I_total); 
 fprintf('%TDL test error= %f. in %f-th day \n',TDL_test_err(test_day),test_day); 

end

figure(1);
plot(HOOI_test_err);
hold on;
plot(TDL_test_err);
legend('HOOI','TDL');
