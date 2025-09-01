rc = DefineNumber[1.4286];
rf = DefineNumber[0.2]; 

xmin = DefineNumber[-10];
xmax = DefineNumber[10];
ymin = DefineNumber[-10];
ymax = DefineNumber[10];

xcmin = DefineNumber[-5];
xcmax = DefineNumber[5];
ycmin = DefineNumber[-0.5];
ycmax = DefineNumber[0.5];

Point(1) = {xmin, ymin, 0, rc};
Point(2) = {xmin, ymax, 0, rc};
Point(3) = {xmax, ymax, 0, rc};
Point(4) = {xmax, ymin, 0, rc};


Point(5) = {xcmin, ycmin, 0, rf};
Point(6) = {xcmin, ycmax, 0, rf};
Point(7) = {xcmax, ycmax, 0, rf};
Point(8) = {xcmax, ycmin, 0, rf};


//+
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 5};

Curve Loop(1) = {1, 2, 3, 4};
Curve Loop(2) = {5, 6, 7, 8};
Plane Surface(1) = {1, 2};

//Physical Surface("Domain", 9) = {3, 4, 1, 2};
//Physical Surface("object", 11) = {5, 6, 7, 8};
