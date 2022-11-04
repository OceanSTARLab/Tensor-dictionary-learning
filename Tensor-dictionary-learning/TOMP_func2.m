
% TOMP :sparse coding with tensor-based OMP
% input£ºtraining tensor T;   dictionary U_tucker£» sparse coefficients(maximum number of non-zeros coefficients) t;
% tolerence 
% output£ºsparse code tensor S

function [S,Non_zeroindex] = TOMP_func2(T,U_tucker,tolerence,t_max)
%[I_1,I_2,I_3]=size(T);
disp('--------------------SPARSE-CORE-TENSOR------TRAINING-------------------------------------')
M_1=size(U_tucker{1},2);
M_2=size(U_tucker{2},2);
M_3=size(U_tucker{3},2);
%tensor stacking 
T_vec=reshape(T,[],1);
A=kron(kron(U_tucker{3},U_tucker{2}),U_tucker{1});
% -- Intitialize --
% start at s = 0,  r = T_vec - A*s = T_vec
r_n          = T_vec;            % Initial residual
normR       = norm(r_n,2);       % Norm of residual
t=1;                             % iteration number 
M = size(A,2);
I = size(A,1);
theta=zeros(M,1);                         
At=zeros(I,1);                            
Pos_theta=zeros(1,1);                    

A_norm=ones(M,1);
for i=1:M
    A_norm(i)=norm( A(:,i));
end


     while  t<=t_max %&&  normR >tolerence         
         product=A'*r_n./A_norm;                       
        [~,pos]=max(abs(product));          
         At(:,t)=A(:,pos);                     
         Pos_theta(t)=pos;                     
         A(:,pos)=zeros(I,1);                                                                  
         theta_ls=pinv((At(:,1:t)'*At(:,1:t)))*At(:,1:t)'*T_vec;                                                         
         normR_last=  normR;                                                      
         r_n=T_vec-At(:,1:t)*theta_ls;             
%         normR= norm(r_n,2); 
%          if  abs(normR_last- normR)<1e-3
%              break
%          end
         t=t+1;
    end
     theta(Pos_theta)=theta_ls;                  


S=reshape(theta,M_1,M_2,M_3);

%Transform Pos_theta index(vector) to tensor-based index
index=Pos_theta;
num=size(index,2);
M1=M_1;M2=M_2;M3=M_3;

M3_index=floor((index-1)./(M1*M2))+1;
index=index-(M3_index-1).*(M1*M2);

M2_index=floor((index-1)./(M1))+1;
index=index-(M2_index-1).*(M1);

M1_index=index;

Non_zeroindex=zeros(num,3);
Non_zeroindex(:,1)=M1_index';
Non_zeroindex(:,2)=M2_index';
Non_zeroindex(:,3)=M3_index';

end