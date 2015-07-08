import numpy as np
import matplotlib.pyplot as plt
import scipy.io

def mind(x,y,z,dincl,ddecl,x0,y0,z0,aincl,adecl):

	x = np.array(x, dtype=float)
	y = np.array(y, dtype=float)
	z = np.array(z, dtype=float)
	x0 = np.array(x0, dtype=float)
	y0 = np.array(y0, dtype=float)
	z0 = np.array(z0, dtype=float)
	dincl = np.array(dincl, dtype=float)
	ddecl = np.array(ddecl, dtype=float)
	aincl = np.array(aincl, dtype=float)
	adecl = np.array(adecl, dtype=float)


	di=np.pi*dincl/180.0
	dd=np.pi*ddecl/180.0

	cx=np.cos(di)*np.cos(dd)
	cy=np.cos(di)*np.sin(dd)
	cz=np.sin(di)


	ai=np.pi*aincl/180.0
	ad=np.pi*adecl/180.0

	ax=np.cos(ai)*np.cos(ad)
	ay=np.cos(ai)*np.sin(ad)
	az=np.sin(ai)


	# begin the calculation
	a=x-x0
	b=y-y0
	h=z-z0

	rt=np.sqrt(a**2.+b**2.+h**2.)**5.

	txy=3.*a*b/rt
	txz=3.*a*h/rt
	tyz=3.*b*h/rt

	txx=(2.*a**2.-b**2.-h**2.)/rt
	tyy=(2.*b**2.-a**2.-h**2.)/rt
	tzz=-(txx+tyy)

	bx= (txx*cx+txy*cy+txz*cz)
	by= (txy*cx+tyy*cy+tyz*cz)
	bz= (txz*cx+tyz*cy+tzz*cz)

	return bx*ax+by*ay+bz*az


def fem3loop(L,R,xc,yc,zc,dincl,ddecl,S,ht,f,xmin,xmax,dx):

	L = np.array(L, dtype=float)
	R = np.array(R, dtype=float)
	xc = np.array(xc, dtype=float)
	yc = np.array(yc, dtype=float)
	zc = np.array(zc, dtype=float)
	dincl = np.array(dincl, dtype=float)
	ddecl = np.array(ddecl, dtype=float)
	S = np.array(S, dtype=float)
	ht = np.array(ht, dtype=float)
	f = np.array(f, dtype=float)
	xmin = np.array(xmin, dtype=float)
	xmax = np.array(xmax, dtype=float)
	dx = np.array(dx, dtype=float)

	ymin = xmin
	ymax = xmax
	dely = dx

	# generate the grid
	xp=np.arange(xmin,xmax,dx)
	yp=np.arange(ymin,ymax,dely)
	[y,x]=np.meshgrid(yp,xp)
	z=0.*x-ht

	# set up the response arrays
	real_response=0.0*x
	imag_response=0.0*x

	# frequency characteristics
	alpha=2.*np.pi*f*L/R

	f_factor=(alpha**2.+1j*alpha)/(1+alpha**2.)

	amin=0.01
	amax=100.
	da=4./40.
	alf=np.arange(-2.,2.,da)
	alf=10.**alf

	fre=alf**2./(1.+alf**2.)
	fim=alf/(1.+alf**2.)


	# simulate anomalies
	yt=y-S/2.
	yr=y+S/2.

	dm=-S/2.
	dp= S/2.

	M13=mind(0.,dm,0.,90.,0., 0., dp, 0., 90.,0.)
	M12=L*mind(x,yt,z,90.,0.,xc,yc,zc,dincl,ddecl)
	M23=L*mind(xc,yc,zc,dincl,ddecl,x,yr,z,90.,0.)

	c_response=-M12*M23*f_factor/(M13*L)

	# scaled to simulate a net volumetric effect
	real_response=np.real(c_response)*1000.
	imag_response=np.imag(c_response)*1000.

	fig, ax = plt.subplots(2,2, figsize = (10,6))

	plt.subplot(2,2,1)
	plt.semilogx(alf,fre,'.-b')
	plt.semilogx(alf,fim,'.--g')
	plt.plot([alpha, alpha],[0., 1.],'-k')
	plt.legend(['Real','Imag'],loc=2)
	plt.xlabel('$\\alpha = \\omega L /R$')
	plt.ylabel('Frequency Response')
	plt.title('Plot 1: EM responses of loop')

	plt.subplot(2,2,2)
	kx = np.ceil(xp.size/2.)
	plt.plot(y[kx,:],real_response[kx,:],'.-b')
	plt.plot(y[kx,:],imag_response[kx,:],'.--g')
	# plt.legend(['Real','Imag'],loc=2)
	plt.xlabel('Easting')
	plt.ylabel('H$_s$/H$_p$')
	plt.title('Plot 2: EW cross section along Northing = %1.1f' %(x[kx,0]))

	vminR = real_response.min()
	vmaxR = real_response.max()
	plt.subplot(2,2,3)
	plt.plot(np.r_[xp.min(),xp.max()], np.zeros(2), 'k--', lw=1)
	plt.imshow(real_response,extent=[xp.min(),xp.max(),yp.min(),yp.max()], vmin = vminR, vmax = vmaxR)

	plt.xlabel('Easting (m)')
	plt.ylabel('Northing (m)')
	plt.title('Plot 3: Real Component')
	clb = plt.colorbar()
	clb.set_label('H$_s$/H$_p$')
	plt.tight_layout()

	vminI = imag_response.min()
	vmaxI = imag_response.max()
	plt.subplot(2,2,4)
	plt.plot(np.r_[xp.min(),xp.max()], np.zeros(2), 'k--', lw=1)
	plt.imshow(imag_response,extent=[xp.min(),xp.max(),yp.min(),yp.max()], vmin = vminI, vmax = vmaxI)

	plt.xlabel('Easting (m)')
	plt.ylabel('Northing (m)')
	plt.title('Plot 4: Imag Component')
	clb = plt.colorbar()
	clb.set_label('H$_s$/H$_p$')
	plt.tight_layout()
	plt.show()

def interactfem3loop(L,R,xc,yc,zc,dincl,ddecl,f,dx,default=False):
	if default == True:
		dx = 0.25
		xc = 0.
		yc = 0.
		zc = 1.
		dincl = 0.
		ddecl = 90.
		L = 0.1
		R = 2000.

	S = 4.
	ht = 1.
	xmin = -40.*dx
	xmax = 40.*dx
	return fem3loop(L,R,-yc,xc,zc,dincl,ddecl,S,ht,f,xmin,xmax,dx)


if __name__ == '__main__':
	L = 0.1
	R = 2000
	xc = 0.
	yc = 0.
	zc = 2.
	dincl = 0.
	ddecl = 90.
	S = 4.
	ht = 0.
	f = 10000.
	xmin = -10.
	xmax = 10.
	dx = 0.25

	fem3loop(L,R,xc,yc,zc,dincl,ddecl,S,ht,f,xmin,xmax,dx)
