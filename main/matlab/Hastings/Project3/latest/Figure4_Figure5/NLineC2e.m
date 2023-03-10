delta = 0.95;
pp = 10;
T = Inf;
S = simplexgrid(8,pp,1);
load NLine.mat;
r1 = e12;
r2 = e21;
r3 = e23;
r4 = e32;
r5 = e34;
r6 = e43;
r7 = e45;
r8 = e54;
r9 = e56;
r10 = e65;
r11 = e67;
r12 = e76;
r13 = e18;
r14 = e81;
r15 = e16;
r16 = e61;
r17 = e27;
r18 = e72;
r19 = e38;
r20 = e83;
r21 = e47;
r22 = e74;
r23 = e58;
r24 = e85;
c1 = 2;
c2 = 25;
n = 6;
len_d = 3;
len_b = 3;
val = zeros(len_d,len_b);
count = 0;
Xvals = [repelem(1:4,1,8)' repmat(1:8,1,4)'];
P1 = [0, 0, 0, 0, 0, 1, 0, 0;
    0, 0, 0, 0, 0, 0, 1, 0;
    0, 0, 0, 1, 0, 0, 0, 0;
    0, 0, 0, 1, 0, 0, 0, 0;
    0, 0, 0, 0, 1, 0, 0, 0;
    0, 0, 0, 0, 0, 1, 0, 0;
    0, 0, 0, 0, 0, 0, 1, 0;
    0, 0, 0, 0, 1, 0, 0, 0];
P3 = [0, 1, 0, 0, 0, 0, 0, 0;
    0, 1, 0, 0, 0, 0, 0, 0;
    0, 0, 1, 0, 0, 0, 0, 0;
    0, 0, 0, 1, 0, 0, 0, 0;
    0, 0, 0, 1, 0, 0, 0, 0;
    0, 0, 0, 0, 0, 0, 1, 0;
    0, 0, 0, 0, 0, 0, 1, 0;
    0, 0, 1, 0, 0, 0, 0, 0];
P4 = [0, 0, 0, 0, 0, 0, 0, 1;
    0, 0, 1, 0, 0, 0, 0, 0;
    0, 0, 1, 0, 0, 0, 0, 0;
    0, 0, 0, 1, 0, 0, 0, 0;
    0, 0, 0, 0, 1, 0, 0, 0;
    0, 0, 0, 0, 1, 0, 0, 0;
    0, 0, 0, 1, 0, 0, 0, 0;
    0, 0, 0, 0, 0, 0, 0, 1];
R = [0 c1*c2 0 0 c2 c2+c1*c2 c2 c2 2*c2 2*c2+c1*c2 2*c2 2*c2 3*c2 3*c2+c1*c2 3*c2 3*c2 2*c2 2*c2+c1*c2 2*c2 2*c2 c2 c2+c1*c2 c2 c2 2*c2 2*c2+c1*c2 2*c2 2*c2 c2 c2+c1*c2 c2 c2]-c1*c2;
opt1 = struct('Qtype',0,'Rtype',2);
opt2 = struct('maxit',600,'nochangelim',500,'prtiters',0,'print',0);
ind_a = 3;
ind_h = 6;
ind_l = 1;
p = 1;
q = 0;
Q1 = [p^2 p*q p*q q^2;
    p*q p^2 q^2 p*q;
    p*q p^2 q^2 p*q;
    q^2 p*q p*q p^2;
    p*q q^2 p^2 p*q;
    p*q q^2 p^2 p*q;
    q^2 p*q p*q p^2;
    p^2 p*q p*q q^2];
Q = repmat(Q1',1,4);

for ind_d = 1:len_d
    for ind_b = 1:len_b
        e12 = r1(ind_d,ind_b);
        e21 = r2(ind_d,ind_b);
        e23 = r3(ind_d,ind_b);
        e32 = r4(ind_d,ind_b);
        e34 = r5(ind_d,ind_b);
        e43 = r6(ind_d,ind_b);
        e45 = r7(ind_d,ind_b);
        e54 = r8(ind_d,ind_b);
        e56 = r9(ind_d,ind_b);
        e65 = r10(ind_d,ind_b);
        e67 = r11(ind_d,ind_b);
        e76 = r12(ind_d,ind_b);
        e18 = r13(ind_d,ind_b);
        e81 = r14(ind_d,ind_b);
        e16 = r15(ind_d,ind_b);
        e61 = r16(ind_d,ind_b);
        e27 = r17(ind_d,ind_b);
        e72 = r18(ind_d,ind_b);
        e38 = r19(ind_d,ind_b);
        e83 = r20(ind_d,ind_b);
        e47 = r21(ind_d,ind_b);
        e74 = r22(ind_d,ind_b);
        e58 = r23(ind_d,ind_b);
        e85 = r24(ind_d,ind_b);
        P2 = [0, e12/(e12+e16+e18), 0, 0, 0, e16/(e12+e16+e18), 0, e18/(e12+e16+e18);
            e21/(e21+e23+e27), 0, e23/(e21+e23+e27), 0, 0, 0, e27/(e21+e23+e27), 0;
            0, e32/(e32+e34+e38), 0, e34/(e32+e34+e38), 0, 0, 0, e38/(e32+e34+e38);
            0, 0, e43/(e43+e45+e47), 0, e45/(e43+e45+e47), 0, e47/(e43+e45+e47), 0;
            0, 0, 0, e54/(e54+e56+e58), 0, e56/(e54+e56+e58), 0, e58/(e54+e56+e58);
            e61/(e61+e65+e67), 0, 0, 0, e65/(e61+e65+e67), 0, e67/(e61+e65+e67), 0;
            0, e72/(e72+e74+e76), 0, e74/(e72+e74+e76), 0, e76/(e72+e74+e76), 0, 0;
            e81/(e81+e83+e85), 0, e83/(e81+e83+e85), 0, e85/(e81+e83+e85), 0, 0, 0];
        P = [P1' P2' P3' P4'];
        [b,Pb,Rb] = pomdp(pp,P,Q,R,opt1);
        model = struct('P',Pb,'R',Rb,'discount',delta,'T',T);
        results = mdpsolve(model,opt2);
        f = results.v;
        val(ind_d,ind_b) = f(S(:,4)==1);
        count = count + 1;
        disp(count)
    end
end
save NLineC2e.mat val;