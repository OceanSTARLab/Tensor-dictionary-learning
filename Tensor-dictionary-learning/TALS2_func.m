
% TALS2 :
% input£ºtraining tensor T; sparse code tensor S; initial dictionary U0_tucker
% output£ºU,SparseCode

function [U,SparseCode]= TALS2_func(T,S,U0_tucker,n_nonzero_coefs)
% %[U]= TALS_func(Y,coretensor,U0_tucker)
% T=Y;
% S=coretensor;

 disp('--------------------DICTIONARY------TRAINING-------------------------------------')
 max_it=100;
 tolerance=1e-10 ;
 U=U0_tucker;
 SparseCode=S;

%step1 update for U{1}
%only update the chosen dictionary
T_1=tenmat(T,1) ;
S_1=tenmat(SparseCode,1) ;
zero_column=sum(S_1.data');
[~,index_m1]=find(zero_column~=0);
temp=S_1.data*(kron(U{3},U{2})' );
temp_pop1=temp(index_m1,:);
U{1}(:,index_m1)=T_1.data*pinv(temp_pop1) ;
[SparseCode,~] = TOMP_func2(T,U, tolerance,n_nonzero_coefs);
%step2 update for U{2}
T_2=tenmat(T,2) ;
S_2=tenmat(SparseCode,2) ;
zero_column2=sum(S_2.data');
[~,index_m2]=find(zero_column2~=0);
temp=S_2.data*(kron(U{3},U{1})' );
temp_pop2=temp(index_m2,:);
U{2}(:,index_m2)=T_2.data*pinv(temp_pop2) ;
[SparseCode,~] = TOMP_func2(T,U, tolerance,n_nonzero_coefs);
%step3 update for U{3}
T_3=tenmat(T,3) ;
S_3=tenmat(SparseCode,3) ;
zero_column3=sum(S_3.data');
[~,index_m3]=find(zero_column3~=0);
temp=S_3.data*(kron(U{2},U{1})' );
temp_pop3=temp(index_m3,:);
U{3}(:,index_m3)=T_3.data*pinv(temp_pop3) ;
[SparseCode,~] = TOMP_func2(T,U, tolerance,n_nonzero_coefs);



%T_recon=lmlragen(U,SparseCode);  
% if   norm(T(:)-T_recon(:))   < tolerance
%     break
% end


end