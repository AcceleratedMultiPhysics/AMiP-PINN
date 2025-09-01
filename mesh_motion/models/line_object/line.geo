//+

rc = DefineNumber[1.4286];
rf = DefineNumber[0.2]; 
Point(1) = {-10, -10, 0, rc};
Point(2) = {10, -10, 0, rc};
Point(3) = {10, 10, 0, rc};
Point(4) = {-10, 10, 0, rc};


Point(5) = {0, -10, 0, rc};
Point(6) = {10, 0, 0, rc};
Point(7) = {0, 10, 0, rc};
Point(8) = {-10, 0, 0, rc};

Point(11) = {0, 0, 0, rf};
Point(12) = {5, 0, 0, rf};
Point(13) = {-5, 0, 0, rf};

//Point(14) = {-5, -0.2, 0, rf};
//Point(15) = {-5, 0.2, 0, rf};
//Point(16) = {5, 0.2, 0, rf};
//Point(17) = {5, -0.2, 0, rf};

//+
Line(1) = {1, 5};
Line(2) = {5, 2};
Line(3) = {2, 6};
Line(4) = {6, 3};
Line(5) = {3, 7};
Line(6) = {7, 4};
Line(7) = {4, 8};
Line(8) = {8, 1};

Line(9)  = {11, 5};
Line(10) = {11, 12};
Line(11) = {12, 6};
Line(12) = {11, 7};
Line(13) = {11, 13};
Line(14) = {13, 8};

//Line(15) = {14, 15};
//Line(16) = {15, 16};
//Line(17) = {16, 17};
//Line(18) = {17, 14};

Curve Loop(1) = {12, -5, -4, -11, -10};
Plane Surface(1) = {1};
Curve Loop(2) = {13, 14, -7, -6, -12};
Plane Surface(2) = {2};
Curve Loop(3) = {1, -9, 13, 14, 8};
Plane Surface(3) = {3};
Curve Loop(4) = {2, 3, -11, -10, 9};
Plane Surface(4) = {4};

Physical Surface("Domain", 9) = {3, 4, 1, 2};
Physical Line("object", 9) = {13, 11, 12};

