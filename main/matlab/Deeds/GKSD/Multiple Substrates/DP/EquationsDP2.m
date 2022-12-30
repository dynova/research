function ydot = EquationsDP2(t,y)
global Q1 Q2 delta1 delta2 Moff Mon Dcat l Doff Don Mcat
S1 = y(1:l+1);
MS1 = y(l+2:2*l+1);
DS1 = y(2*l+2:3*l+1);
S2 = y(3*l+2:4*l+2);
MS2 = y(4*l+3:5*l+2);
DS2 = y(5*l+3:6*l+2);
M = y(6*l+3);
D = y(6*l+4);
ydot = zeros(6*l+4,1);
firstrunsumS1 = sum(S1(1:l));
secondrunsumS1 = sum(S1(2:l+1));
runsumMS1 = sum(MS1(5:l));
runsumDS1 = sum(DS1(4:l));
firstrunsumS2 = sum(S2(1:l));
secondrunsumS2 = sum(S2(2:l+1));
runsumMS2 = sum(MS2(5:l));
runsumDS2 = sum(DS2(4:l));
ydot(1) = Q1+Moff*MS1(1)-(Mon*M+delta1)*S1(1)+Dcat*DS1(1);
ydot(2) = Moff*MS1(2)+Doff*DS1(1)-(Mon*M+Don*D+delta1)*S1(2)+Mcat*MS1(1);
ydot(3) = Moff*MS1(3)+Doff*DS1(2)-(Mon*M+Don*D+delta1)*S1(3)+Mcat*MS1(2);
ydot(4) = Moff*MS1(4)+Doff*DS1(3)-(Mon*M+Don*D+delta1)*S1(4)+Mcat*MS1(3);
ydot(5:l) = Moff*(MS1(5:l))+Doff*(DS1(4:l-1))-(Mon*M+Don*D+delta2)*S1(5:l)+Mcat*MS1(4:l-1);
ydot(l+1) = Doff*DS1(l)-(Don*D+delta2)*S1(l+1)+Mcat*MS1(l);
ydot(l+2) = Mon*M*S1(1)-(Moff+Mcat+delta1)*MS1(1);
ydot(l+3) = Mon*M*S1(2)-(Moff+Mcat+delta1)*MS1(2);
ydot(l+4) = Mon*M*S1(3)-(Moff+Mcat+delta1)*MS1(3);
ydot(l+5) = Mon*M*S1(4)-(Moff+Mcat+delta1)*MS1(4);
ydot(l+6:2*l+1) = Mon*M*S1(5:l)-(Moff+Mcat+delta2)*MS1(5:l);
ydot(2*l+2) = Dcat*DS1(2)+Don*D*S1(2)-(Doff+Dcat+delta1)*DS1(1);
ydot(2*l+3) = Dcat*DS1(3)+Don*D*S1(3)-(Doff+Dcat+delta1)*DS1(2);
ydot(2*l+4) = Dcat*DS1(4)+Don*D*S1(4)-(Doff+Dcat+delta1)*DS1(3);
ydot(2*l+5) = Dcat*DS1(5)+Don*D*S1(5)-(Doff+Dcat+delta2)*DS1(4);
ydot(2*l+6:3*l) = Dcat*DS1(6:l)+Don*D*S1(6:l)-(Doff+Dcat+delta2)*DS1(5:l-1);
ydot(3*l+1) = Don*D*S1(l+1)-(Doff+Dcat+delta2)*DS1(l);
ydot(3*l+2) = Q2+Moff*MS2(1)-(Mon*M+delta1)*S2(1)+Dcat*DS2(1);
ydot(3*l+3) = Moff*MS2(2)+Doff*DS2(1)-(Mon*M+Don*D+delta1)*S2(2)+Mcat*MS2(1);
ydot(3*l+4) = Moff*MS2(3)+Doff*DS2(2)-(Mon*M+Don*D+delta1)*S2(3)+Mcat*MS2(2);
ydot(3*l+5) = Moff*MS2(4)+Doff*DS2(3)-(Mon*M+Don*D+delta1)*S2(4)+Mcat*MS2(3);
ydot(3*l+6:4*l+1) = Moff*(MS2(5:l))+Doff*(DS2(4:l-1))-(Mon*M+Don*D+delta2)*S2(5:l)+Mcat*MS2(4:l-1);
ydot(4*l+2) = Doff*DS2(l)-(Don*D+delta2)*S2(l+1)+Mcat*MS2(l);
ydot(4*l+3) = Mon*M*S2(1)-(Moff+Mcat+delta1)*MS2(1);
ydot(4*l+4) = Mon*M*S2(2)-(Moff+Mcat+delta1)*MS2(2);
ydot(4*l+5) = Mon*M*S2(3)-(Moff+Mcat+delta1)*MS2(3);
ydot(4*l+6) = Mon*M*S2(4)-(Moff+Mcat+delta1)*MS2(4);
ydot(4*l+7:5*l+2) = Mon*M*S2(5:l)-(Moff+Mcat+delta2)*MS2(5:l);
ydot(5*l+3) = Dcat*DS2(2)+Don*D*S2(2)-(Doff+Dcat+delta1)*DS2(1);
ydot(5*l+4) = Dcat*DS2(3)+Don*D*S2(3)-(Doff+Dcat+delta1)*DS2(2);
ydot(5*l+5) = Dcat*DS2(4)+Don*D*S2(4)-(Doff+Dcat+delta1)*DS2(3);
ydot(5*l+6) = Dcat*DS2(5)+Don*D*S2(5)-(Doff+Dcat+delta2)*DS2(4);
ydot(5*l+7:6*l+1) = Dcat*DS2(6:l)+Don*D*S2(6:l)-(Doff+Dcat+delta2)*DS2(5:l-1);
ydot(6*l+2) = Don*D*S2(l+1)-(Doff+Dcat+delta2)*DS2(l);
ydot(6*l+3) = (Moff+Mcat+delta1)*(MS1(1)+MS1(2)+MS1(3)+MS1(4)+MS2(1)+MS2(2)+MS2(3)+MS2(4))+...
              (Moff+Mcat+delta2)*(runsumMS1+runsumMS2)-Mon*M*(firstrunsumS1+firstrunsumS2);
ydot(6*l+4) = (Doff+delta1)*(DS1(1)+DS1(2)+DS1(3)+DS2(1)+DS2(2)+DS2(3))+...
              (Doff+delta2)*(runsumDS1+runsumDS2)-Don*D*(secondrunsumS1+secondrunsumS2)+...
              Dcat*DS1(1);
end