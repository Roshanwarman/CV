from nsimulator import nsimulator


double = nsimulator(2)

double.set_mpotential(0,-20)

double.set_mpotential(1,-80)

double.set_Ie(0, 2.5)
double.set_Ie(1,2.5)

double.make_inhibitory_connection(0,1)
# double.make_inhibitory_connection(1,0)

double.simulate_timeinterval(1,.1)

double.plot_membranepotentials([0 , 1])
