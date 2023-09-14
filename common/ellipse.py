# from https://github.com/RSKothari/EllSeg/blob/master/helperfunctions.py
import numpy as np

def rotation_2d(theta):
    # Return a 2D rotation matrix in the anticlockwise direction
    c, s = np.cos(theta), np.sin(theta)
    H_rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1]])
    return H_rot

def trans_2d(cx, cy):
    H_trans = np.array([[1.0, 0.0, cx], [0.0, 1.0, cy], [0.0, 0.0, 1]])
    return H_trans

def scale_2d(sx, sy):
    H_scale = np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1]])
    return H_scale

EPS = 1e-40
class my_ellipse():
    def __init__(self, param):
        '''
        Accepts parameterized form
        '''
        self.EPS = 1e-3
        if param is not list:
            self.param = param
            self.mat = self.param2mat(self.param)
            self.quad = self.mat2quad(self.mat)
            #self.Phi = self.recover_Phi()

    def param2mat(self, param):
        cx, cy, a, b, theta = tuple(param)
        H_rot = rotation_2d(-theta)
        H_trans = trans_2d(-cx, -cy)

        A, B = 1/a**2, 1/b**2
        Q = np.array([[A, 0, 0], [0, B, 0], [0, 0, -1]])
        mat = H_trans.T @ H_rot.T @ Q @ H_rot @ H_trans
        return mat

    def mat2quad(self, mat):
        assert np.sum(np.abs(mat.T - mat)) <= self.EPS, 'Conic form incorrect'
        a, b, c, d, e, f = mat[0,0], 2*mat[0, 1], mat[1,1], 2*mat[0, 2], 2*mat[1, 2], mat[-1, -1]
        return np.array([a, b, c, d, e, f])

    def quad2param(self, quad):
        mat = self.quad2mat(quad)
        param = self.mat2param(mat)
        return param

    def quad2mat(self, quad):
        a, b, c, d, e, f = tuple(quad)
        mat = np.array([[a, b/2, d/2], [b/2, c, e/2], [d/2, e/2, f]])
        return mat

    def mat2param(self, mat):
        assert np.sum(np.abs(mat.T - mat)) <= self.EPS, 'Conic form incorrect'
        # Estimate rotation
        theta = self.recover_theta(mat)
        # Estimate translation
        tx, ty = self.recover_C(mat)
        # Invert translation and rotation
        H_rot = rotation_2d(theta)
        H_trans = trans_2d(tx, ty)
        mat_norm = H_rot.T @ H_trans.T @ mat @ H_trans @ H_rot
        major_axis = np.sqrt(1/mat_norm[0,0])
        minor_axis = np.sqrt(1/mat_norm[1,1])
        area = np.pi*major_axis*minor_axis
        return np.array([tx, ty, major_axis, minor_axis, theta, area])

    def phi2param(self, xm, ym):
        '''
        Given phi values, compute ellipse parameters

        Parameters
        ----------
        Phi : np.array [5, ]
            for information on Phi values, please refer to ElliFit.
        xm : int
        ym : int

        Returns
        -------
        param : np.array [5, ].
            Ellipse parameters, [cx, cy, a, b, theta]

        '''
        try:
            x0=(self.Phi[2]-self.Phi[3]*self.Phi[1])/((self.Phi[0])-(self.Phi[1])**2)
            y0=(self.Phi[0]*self.Phi[3]-self.Phi[2]*self.Phi[1])/((self.Phi[0])-(self.Phi[1])**2)
            term2=np.sqrt(((1-self.Phi[0])**2+4*(self.Phi[1])**2))
            term3=(self.Phi[4]+(y0)**2+(x0**2)*self.Phi[0]+2*self.Phi[1])
            term1=1+self.Phi[0]
            print(term1, term2, term3)
            b=(np.sqrt(2*term3/(term1+term2)))
            a=(np.sqrt(2*term3/(term1-term2)))
            alpha=0.5*np.arctan2(2*self.Phi[1],1-self.Phi[0])
            model = [x0+xm, y0+ym, a, b, -alpha]
        except:
            print('Inappropriate model generated')
            model = [np.nan, np.nan, np.nan, np.nan, np.nan]
        if np.all(np.isreal(model)) and np.all(~np.isnan(model)) and np.all(~np.isinf(model)):
            model = model
        else:
            model = [-1, -1, -1, -1, -1]
        return model

    def recover_theta(self, mat):
        a, b, c, d, e, f = tuple(self.mat2quad(mat))
        #print('a: {}. b: {}. c: {}'.format(a, b, c))
        if abs(b)<=EPS and a<=c:
            theta = 0.0
        elif abs(b)<=EPS and a>c:
            theta=np.pi/2
        elif abs(b)>EPS and a<=c:
            theta=0.5*np.arctan2(b, (a-c))
        elif abs(b)>EPS and a>c:
            #theta = 0.5*(np.pi + np.arctan(b/(a-c)))
            theta = 0.5*np.arctan2(b, (a-c))
        else:
            print('Unknown condition')
        return theta

    def recover_C(self, mat):
        a, b, c, d, e, f = tuple(self.mat2quad(mat))
        tx = (2*c*d - b*e)/(b**2 - 4*a*c)
        ty = (2*a*e - b*d)/(b**2 - 4*a*c)
        return (tx, ty)

    def transform(self, H):
        '''
        Given a transformation matrix H, modify the ellipse
        '''
        mat_trans = np.linalg.inv(H.T) @ self.mat @ np.linalg.inv(H)
        return self.mat2param(mat_trans), self.mat2quad(mat_trans), mat_trans

    def verify(self, pts):
        '''
        Given an array of points Nx2, verify the ellipse model
        '''
        N = pts.shape[0]
        pts = np.concatenate([pts, np.ones((N, 1))], axis=1)
        err = 0.0
        for i in range(0, N):
            err+=pts[i, :]@self.mat@pts[i, :].T # Note that the transpose here is irrelevant
        return np.inf if (N==0) else err/N

    def generatePoints(self, N, mode):
        '''
        Generates 8 points along the periphery of an ellipse. The mode dictates
        the uniformity between points.
        mode: str
        'equiAngle' - Points along the periphery with angles [0:45:360)
        'equiSlope' - Points along the periphery with tangential slopes [-1:0.5:1)
        'random' - Generate N points randomly across the ellipse
        '''
        from itertools import chain

        a = self.param[2]
        b = self.param[3]

        alpha = (a*np.sin(self.param[-1]))**2 + (b*np.cos(self.param[-1]))**2
        beta = (a*np.cos(self.param[-1]))**2 + (b*np.sin(self.param[-1]))**2
        gamma = (a**2 - b**2)*np.sin(2*self.param[-1])

        if mode == 'equiSlope':
            slope_list = [1e-6, 1, 1000, -1]
            K_fun = lambda m_i:  (m_i*gamma + 2*alpha)/(2*beta*m_i + gamma)

            x_2 = [((a*b)**2)/(alpha + beta*K_fun(m)**2 - gamma*K_fun(m)) for m in slope_list]

            x = [(+np.sqrt(val), -np.sqrt(val)) for val in x_2]
            y = []
            for i, m in enumerate(slope_list):
                y1 = -x[i][0]*K_fun(m)
                y2 = -x[i][1]*K_fun(m)
                y.append((y1, y2))
            y_r = np.array(list(chain(*y))) + self.param[1]
            x_r = np.array(list(chain(*x))) + self.param[0]

        if mode == 'equiAngle':

            T = 0.5*np.pi*np.array([-1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])
            N = len(T)
            x = self.param[2]*np.cos(T)
            y = self.param[3]*np.sin(T)
            H_rot = rotation_2d(self.param[-1])
            X1 = H_rot.dot(np.stack([x, y, np.ones(N, )], axis=0))

            x_r = X1[0, :] + self.param[0]
            y_r = X1[1, :] + self.param[1]

        elif mode == 'random':
            T = 2*np.pi*(np.random.rand(N, ) - 0.5)
            x = self.param[2]*np.cos(T)
            y = self.param[3]*np.sin(T)
            H_rot = rotation_2d(self.param[-1])
            X1 = H_rot.dot(np.stack([x, y, np.ones(N, )], axis=0))
            x_r = X1[0, :] + self.param[0]
            y_r = X1[1, :] + self.param[1]

        else:
            print('Mode is not defined')

        return x_r, y_r