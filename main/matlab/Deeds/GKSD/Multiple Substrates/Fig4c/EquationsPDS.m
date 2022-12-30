function ydot = EquationsPDS(t,y)
global Q1 Q2 delta1 delta2 Doff Don Dcat Mcat1 Mcat2 l Mon Moff
S1 = y(1:l+1);
MS1 = y(l+2:2*l+2);
DS1 = y(2*l+3:3*l+2);
S2 = y(3*l+3:4*l+3);
MS2 = y(4*l+4:5*l+4);
DS2 = y(5*l+5:6*l+4);
M = y(6*l+5);
D = y(6*l+6);
ydot = zeros(6*l+6,1);
firstrunsumS1 = sum(S1(1:l));
secondrunsumS1 = sum(S1(2:l+1));
runsumMS1 = sum(MS1(5:l));
runsumDS1 = sum(DS1(5:l));
firstrunsumS2 = sum(S2(1:l));
secondrunsumS2 = sum(S2(2:l+1));
runsumMS2 = sum(MS2(5:l));
runsumDS2 = sum(DS2(5:l));
ydot(1) = Q1+Moff*MS1(1)-(Mon*M+delta1)*S1(1)+Dcat*DS1(1);
ydot(2) = Moff*MS1(2)+Doff*DS1(1)-(Mon*M+Don*D+delta1)*S1(2)+Dcat*DS1(2);
ydot(3) = Moff*MS1(3)+Doff*DS1(2)-(Mon*M+Don*D+delta1)*S1(3)+Dcat*DS1(3);
ydot(4) = Moff*MS1(4)+Doff*DS1(3)-(Mon*M+Don*D+delta1)*S1(4)+Dcat*DS1(4);
ydot(5:l) = Moff*MS1(5:l)+Doff*DS1(4:l-1)-(Mon*M+Don*D+delta2)*S1(5:l)+Dcat*DS1(5:l);
ydot(l+1) = Doff*DS1(l)-(Don*D+delta2)*S1(l+1);
ydot(l+2) = Mon*M*S1(1)-(Moff+Mcat1+delta1)*MS1(1);
ydot(l+3) = Mcat1*MS1(1)+Mon*M*S1(2)-(Moff+Mcat2+delta1)*MS1(2);
ydot(l+4) = Mcat2*MS1(2)+Mon*M*S1(3)-(Moff+Mcat2+delta1)*MS1(3);
ydot(l+5) = Mcat2*MS1(3)+Mon*M*S1(4)-(Moff+Mcat2+delta1)*MS1(4);
ydot(l+6:2*l+1) = Mcat2*MS1(4:l-1)+Mon*M*S1(5:l)-(Moff+Mcat2+delta2)*MS1(5:l);
ydot(2*l+2) = Mcat2*MS1(l)-delta2*MS1(l+1);
ydot(2*l+3) = Don*D*S1(2)-(Doff+Dcat+delta1)*DS1(1);
ydot(2*l+4) = Don*D*S1(3)-(Doff+Dcat+delta1)*DS1(2);
ydot(2*l+5) = Don*D*S1(4)-(Doff+Dcat+delta1)*DS1(3);
ydot(2*l+6) = Don*D*S1(5)-(Doff+Dcat+delta1)*DS1(4);
ydot(2*l+7:3*l+1) = Don*D*S1(6:l)-(Doff+Dcat+delta2)*DS1(5:l-1);
ydot(3*l+2) = Don*D*S1(l+1)-(Doff+Dcat+delta2)*DS1(l);
ydot(3*l+3) = Q2+Moff*MS2(1)-(Mon*M+delta1)*S2(1)+Dcat*DS2(1);
ydot(3*l+4) = Moff*MS2(2)+Doff*DS2(1)-(Mon*M+Don*D+delta1)*S2(2)+Dcat*DS2(2);
ydot(3*l+5) = Moff*MS2(3)+Doff*DS2(2)-(Mon*M+Don*D+delta1)*S2(3)+Dcat*DS2(3);
ydot(3*l+6) = Moff*MS2(4)+Doff*DS2(3)-(Mon*M+Don*D+delta1)*S2(4)+Dcat*DS2(4);
ydot(3*l+7:4*l+2) = Moff*MS2(5:l)+Doff*DS2(4:l-1)-(Mon*M+Don*D+delta2)*S2(5:l)+Dcat*DS2(5:l);
ydot(4*l+3) = Doff*DS2(l)-(Don*D+delta2)*S2(l+1);
ydot(4*l+4) = Mon*M*S2(1)-(Moff+Mcat1+delta1)*MS2(1);
ydot(4*l+5) = Mcat1*MS2(1)+Mon*M*S2(2)-(Moff+Mcat2+delta1)*MS2(2);
ydot(4*l+6) = Mcat2*MS2(2)+Mon*M*S2(3)-(Moff+Mcat2+delta1)*MS2(3);
ydot(4*l+7) = Mcat2*MS2(3)+Mon*M*S2(4)-(Moff+Mcat2+delta1)*MS2(4);
ydot(4*l+8:5*l+3) = Mcat2*MS2(4:l-1)+Mon*M*S2(5:l)-(Moff+Mcat2+delta2)*MS2(5:l);
ydot(5*l+4) = Mcat2*MS2(l)-delta2*MS2(l+1);
ydot(5*l+5) = Don*D*S2(2)-(Doff+Dcat+delta1)*DS2(1);
ydot(5*l+6) = Don*D*S2(3)-(Doff+Dcat+delta1)*DS2(2);
ydot(5*l+7) = Don*D*S2(4)-(Doff+Dcat+delta1)*DS2(3);
ydot(5*l+8) = Don*D*S2(5)-(Doff+Dcat+delta1)*DS2(4);
ydot(5*l+9:6*l+3) = Don*D*S2(6:l)-(Doff+Dcat+delta2)*DS2(5:l-1);
ydot(6*l+4) = Don*D*S2(l+1)-(Doff+Dcat+delta2)*DS2(l);
ydot(6*l+5) = (Moff+delta1)*(MS1(1)+MS1(2)+MS1(3)+MS1(4)+MS2(1)+MS2(2)+MS2(3)+MS2(4))+...
    (Moff+delta2)*(runsumMS1+runsumMS2)-Mon*M*(firstrunsumS1+firstrunsumS2)+delta2*(MS1(l+1)+MS2(l+1));
ydot(6*l+6) = (Doff+Dcat+delta1)*(DS1(1)+DS1(2)+DS1(3)+DS1(4)+DS2(1)+DS2(2)+DS2(3)+DS2(4))+...
    (Doff+Dcat+delta2)*(runsumDS1+runsumDS2)-Don*D*(secondrunsumS1+secondrunsumS2);
end