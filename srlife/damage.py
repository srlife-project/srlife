# pylint: disable=no-member
"""
  Module with methods for calculating creep-fatigue damage given
  completely-solved tube results and damage material properties
"""
import numpy as np
import numpy.linalg as la
import scipy.optimize as opt
import multiprocess


class WeibullFailureModel:
  """
    Parent class for time independent Weibull models
  """
  def __init__(self, pset, *args, **kwargs):
    pass

  def calculate_principal_stress(self, stress):
    """
      Calculate the principal stresses given the Mandel vector
    """
    tensor = np.zeros(stress.shape[:2] + (3,3)) #[:1] when no time steps involved
    # indices where (0,0) => (1,1)
    inds = [[(0,0)],[(1,1)],[(2,2)],[(1,2),(2,1)],[(0,2),(2,0)],[(0,1),(1,0)]]
    # multiplicative factors
    mults = [1.0, 1.0, 1.0, np.sqrt(2), np.sqrt(2), np.sqrt(2)]

    for i,(grp, m) in enumerate(zip(inds, mults)):
      for a,b in grp:
        tensor[...,a,b] = stress[...,i] / m

    return la.eigvalsh(tensor)

  def determine_life(self, receiver, material, nthreads = 1,
      decorator = lambda x, n: x):
    """
      Determine the life of the receiver by calculating individual
      material point damage and finding the minimum of all points.

    Determines principal stresses from mandel stress

    Determines tube reliability and overall reliability by taking input of
    element log reliabilities from respective Weibull failure model
    """
    with multiprocess.Pool(nthreads) as p:
      p_tube = list(decorator(
        p.imap(lambda x: self.tube_log_reliability(x, material, receiver), receiver.tubes),
        receiver.ntubes))

    p_tube = np.array(p_tube)

    # Tube reliability is the minimum of all the time steps
    tube = np.min(p_tube, axis = 1)

    # Overall reliability is the minimum of the sum
    overall = np.min(np.sum(p_tube, axis = 0))

    # Convert back from log-prob as we go
    return {"tube_reliability": np.exp(tube), "overall_reliability": np.exp(overall)}


  def tube_log_reliability(self, tube, material, receiver):
    """
    Parent class for crack shape independent models
    which include only PIA and WNTSA models

    Determines normal stress acting on crack
    """

    def __init__(self, pset, *args, **kwargs):
        """
        Create a mesh grid of angles that can represent crack orientations
        Evaluate direction cosines using the angles

        Default values given to nalpha and nbeta which are the number of
        segments in the mesh grid
        """
        super().__init__(pset, *args, **kwargs)

        # limits and number of segments for angles
        self.nalpha = pset.get_default("nalpha", 21)
        self.nbeta = pset.get_default("nbeta", 31)

        # Mesh grid of the vectorized angle values
        self.A, self.B = np.meshgrid(
            np.linspace(0, np.pi, self.nalpha),
            np.linspace(0, 2 * np.pi, self.nbeta, endpoint=False),
            indexing="ij",
        )

    # Add as an element field
    tube.add_quadrature_results("log_reliability",
        np.transpose(np.stack((inc_prob,inc_prob)), axes = (1,2,0)))

        # Direction cosines
        self.l = np.cos(self.A)
        self.m = np.sin(self.A) * np.cos(self.B)
        self.n = np.sin(self.A) * np.sin(self.B)

    def calculate_normal_stress(self, mandel_stress):
        """
        Use direction cosines to calculate the normal stress to
        a crack given the Mandel vector
        """
        # Principal stresses
        pstress = self.calculate_principal_stress(mandel_stress)

        # Normal stress
        return (
            pstress[..., 0, None, None] * (self.l**2)
            + pstress[..., 1, None, None] * (self.m**2)
            + pstress[..., 2, None, None] * (self.n**2)
        )


class CrackShapeDependent(WeibullFailureModel):
    """
    Parent class for crack shape dependent models

    Determines normal, shear, total and equivanlent stresses acting on cracks

    Calculates the element reliability using the equivalent stress from the
    crack shape dependent models
    """

class PIAModel(WeibullFailureModel):
  """
    Principal of independent action failure model
  """

        # Temperature average values
        mavg = np.mean(mvals, axis=0)
        kavg = np.mean(kvals, axis=0)
        Nvavg = np.mean(Nv, axis=0)
        Bvavg = np.mean(Bv, axis=0)

        # Only tension
        pstress[pstress < 0] = 0

        # Max principal stresss over all time steps for each element
        pstress_max = np.max(pstress, axis=0)

        # Calculating ratio of cyclic stress to max cyclic stress (in one cycle)
        # for time-independent and time-dependent cases
        if np.all(time == 0):
            pstress_0 = pstress
        else:
            g = (
                np.trapz(
                    (pstress / (pstress_max + 1.0e-14)) ** Nvavg[..., None],
                    time,
                    axis=0,
                )
                / time[-1]
            )

            # Time dependent principal stress
            pstress_0 = (
                (pstress_max ** Nvavg[..., None] * g * tot_time) / Bvavg[..., None]
                + (pstress_max ** (Nvavg[..., None] - 2))
            ) ** (1 / (Nvavg[..., None] - 2))

        return -kavg * np.sum(pstress_0 ** mavg[..., None], axis=-1) * volumes


class WNTSAModel(CrackShapeIndependent):
    """
    Weibull normal tensile average failure model

    Evaluates an average normal tensile stress (acting on the cracks) integrated over the
    area of a sphere, and uses it to calculate the element reliability
    """

    def calculate_avg_normal_stress(
        self, time, mandel_stress, temperatures, material, tot_time
    ):
        """
        Calculate the average normal tensile stresses from the pricipal stresses

    svals = material.strength(temperatures)
    mvals = material.modulus(temperatures)
    kvals = svals**(-mvals)

    return -kvals * np.sum(pstress**mvals[...,None], axis = -1) * volumes

class WNTSAModel(WeibullFailureModel):
  """
    Weibull normal tensile average failure model

    Assigning default values for nalpha and nbeta
  """

  def __init__(self, pset, *args, **kwargs):
    super().__init__(pset, *args, **kwargs)
    # limits and number of segments for angles
    self.nalpha = pset.get_default("nalpha",21)
    self.nbeta = pset.get_default("nbeta", 31)

  def calculate_avg_normal_stress(self, mandel_stress, mvals, nalpha, nbeta):
    """
        Calculate the average normal tensile stresses given the pricipal stresses
    """
    # Mesh grid of the vectorized angle values
    A,B = np.meshgrid(np.linspace(0,np.pi,self.nalpha),
    np.linspace(0,2*np.pi,self.nbeta,endpoint=False),indexing='ij')

    # Increment of angles to be used in evaluating integral
    dalpha = (A[-1,-1] - A[0,0])/(self.nalpha - 1)
    dbeta = (B[-1,-1] - B[0,0])/(self.nbeta - 1)

    # Direction cosines
    l = np.cos(A)
    m = np.sin(A)*np.cos(B)
    n = np.sin(A)*np.sin(B)

    # Principal stresses
    pstress = self.calculate_principal_stress(mandel_stress)

    # Normal stress
    sigma_n = pstress[...,0,None,None]*(l**2) + pstress[...,1,None,None]*(m**2) + \
    pstress[...,2,None,None]*(n**2)

    # Area integral; suppressing warning given when negative numbers are raised to rational numbers
    with np.errstate(invalid='ignore'):
      integral = ((sigma_n**mvals[...,None,None])*np.sin(A)*dalpha*dbeta)/(4*np.pi)

    # Flatten the last axis and calculate the mean of the positive values along that axis
    flat = integral.reshape(integral.shape[:2] + (-1,))  #[:1] when no time steps involved

    # Suppressing warning to ignoring initial NaN values
    with np.errstate(invalid='ignore'):
    # Average stress
      return np.nansum(np.where(flat >= 0.0,flat,np.nan),axis = -1)


  def calculate_element_log_reliability(self, mandel_stress, temperatures, volumes, material):
    """
      Calculate the element log reliability

      Parameters:
        mandel_stress:  element stresses in Mandel convention
        temperatures:   element temperatures
        volumes:        element volumes
        material:       material model  object with required data
    """

    svals = material.strength(temperatures)
    mvals = material.modulus(temperatures)

    # Average normal tensile stress raied to exponent mv
    avg_nstress = self.calculate_avg_normal_stress(mandel_stress, mvals, self.nalpha, self.nbeta)

    kvals = svals**(-mvals)
    kpvals = (2*mvals + 1)*kvals

    return -kpvals*(avg_nstress)*volumes

class DamageCalculator:
  """
    Parent class for all damage calculators, handling common iteration
    and scaling options
  """
  def __init__(self, pset):
    """
      Parameters:
        pset:       damage parameters
    """
    Maximum tensile stess failure model with a Griffith flaw

    Evaluates an equivalent stress (acting on the cracks) using the normal and
    shear stresses in the maximum tensile stress fracture criterion for a Griffith flaw
    """

      Parameters:
        tube:       fully-populated tube object
        material:   damage material
        receiver:   receiver object (for metadata)
    """
    raise NotImplementedError("Superclass not implemented")

  def determine_life(self, receiver, material, nthreads = 1,
      decorator = lambda x, n: x):
    """
      Determine the life of the receiver by calculating individual
      material point damage and finding the minimum of all points.

        # Projected equivalent stress
        return 0.5 * (sigma_n + np.sqrt((sigma_n**2) + (tau**2)))

    def calculate_kbar(self, temperatures, material, A, dalpha, dbeta):
        """
        Calculate the evaluating normalized crack density coefficient (kbar)
        using the Weibull scale parameter (svals), Weibull modulus (mvals),
        poisson ratio (nu)
        """

        self.temperatures = temperatures
        self.material = material

        # Weibull modulus
        mvals = self.material.modulus(temperatures)

        # Temperature average values
        mavg = np.mean(mvals, axis=0)

        # Evaluating integral for kbar
        integral2 = 2 * (
            (
                (
                    0.5
                    * (
                        ((np.cos(self.A)) ** 2)
                        + np.sqrt(
                            ((np.cos(self.A)) ** 4)
                            + (((np.sin(self.A)) ** 2) * ((np.cos(self.A)) ** 2))
                        )
                    )
                )
                ** mavg[..., None, None]
            )
            * np.sin(self.A)
            * self.dalpha
            * self.dbeta
        )
        kbar = np.pi / np.sum(integral2, axis=(1, 2))

        return kbar


class MTSModelPennyShapedFlaw(CrackShapeDependent):
    """
    Maximum tensile stess failure model with a penny shaped flaw

    Evaluates an equivalent stress (acting on the cracks) using the normal and
    shear stresses in the maximum tensile stress fracture criterion for a penny shaped flaw
    """

    def calculate_eq_stress(self, mandel_stress, temperatures, material):
        """
        Calculate the average normal tensile stresses from the normal and shear stresses

        Additional parameter:
        nu:     Poisson ratio
        """

        # Material parameters
        nu = np.mean(material.nu(temperatures), axis=0)

        # Normal stress
        sigma_n = self.calculate_normal_stress(mandel_stress)

        # Shear stress
        tau = self.calculate_shear_stress(mandel_stress)

        # Projected equivalent stress
        return 0.5 * (
            sigma_n
            + np.sqrt((sigma_n**2) + ((tau / (1 - (0.5 * nu[..., None, None]))) ** 2))
        )

    def calculate_kbar(self, temperatures, material, A, dalpha, dbeta):
        """
        Calculate the evaluating normalized crack density coefficient (kbar)

        Parameters:
        temperatures:   element temperatures
        material:       material model object with required data that includes
                        Weibull modulus (mvals), poisson ratio (nu)
        """

        self.temperatures = temperatures
        self.material = material

        # Weibull modulus
        mvals = self.material.modulus(temperatures)

        # Temperature average values
        mavg = np.mean(mvals, axis=0)

        # Material parameters
        nu = np.mean(material.nu(temperatures), axis=0)

        # Evaluating integral for kbar
        integral2 = 2 * (
            (
                (
                    0.5
                    * (
                        ((np.cos(self.A)) ** 2)
                        + np.sqrt(
                            ((np.cos(self.A)) ** 4)
                            + (
                                ((np.sin(2 * self.A)) ** 2)
                                / (2 - (nu[..., None, None] ** 2))
                            )
                        )
                    )
                )
                ** mavg[..., None, None]
            )
            * np.sin(self.A)
            * self.dalpha
            * self.dbeta
        )
        kbar = np.pi / np.sum(integral2, axis=(1, 2))

        return kbar


class CSEModelGriffithFlaw(CrackShapeDependent):
    """
    Coplanar strain energy failure model with a Griffith flaw

    Evaluates an equivalent stress (acting on the cracks) using the normal and
    shear stresses in the coplanar strain energy fracture criterion
    for a Griffith flaw
    """
    # Material point cycle creep damage
    Dc = self.creep_damage(tube, material, receiver)

    # Material point cycle fatigue damage
    Df = self.fatigue_damage(tube, material, receiver)

    nc = receiver.days

    # This is going to be expensive, but I don't see much way around it
    return min(self.calculate_max_cycles(self.make_extrapolate(c),
      self.make_extrapolate(f), material) for c,f in
        zip(Dc.reshape(nc,-1).T, Df.reshape(nc,-1).T))

  def calculate_max_cycles(self, Dc, Df, material, rep_min = 1, rep_max = 1e6):
    """
    Coplanar strain energy failure model with a penny shaped flaw

    Evaluates an equivalent stress (acting on the cracks) using the normal and
    shear stresses in the coplanar strain energy fracture criterion
    for a penny shaped flaw
    """

    def calculate_eq_stress(self, mandel_stress, temperatures, material):
        """
        Calculate the equivalent stresses from the normal and shear stresses

        Additional parameter:
        nu:     Poisson ratio
        """

        # Material parameters
        nu = np.mean(material.nu(temperatures), axis=0)

        # Normal stress
        sigma_n = self.calculate_normal_stress(mandel_stress)

        # Shear stress
        tau = self.calculate_shear_stress(mandel_stress)

        # Projected equivalent stress
        return np.sqrt((sigma_n**2) + (tau / (1 - (0.5 * nu[..., None, None]))) ** 2)

    def calculate_kbar(self, temperatures, material, A, dalpha, dbeta):
        """
        Calculate the evaluating normalized crack density coefficient (kbar)

        Parameters:
        temperatures:   element temperatures
        material:       material model object with required data that includes
                        Weibull modulus (mvals), poisson ratio (nu)
        """

        self.temperatures = temperatures
        self.material = material

        # Weibull modulus
        mvals = self.material.modulus(temperatures)

        # Temperature average values
        mavg = np.mean(mvals, axis=0)

        # Material parameters
        nu = np.mean(material.nu(temperatures), axis=0)

        # Evaluating integral for kbar
        integral2 = 2 * (
            (
                (
                    np.sqrt(
                        ((np.cos(self.A)) ** 4)
                        + (
                            ((np.sin(2 * self.A)) ** 2)
                            / ((2 - nu[..., None, None]) ** 2)
                        )
                    )
                )
                ** mavg[..., None, None]
            )
            * np.sin(self.A)
            * self.dalpha
            * self.dbeta
        )
        kbar = np.pi / np.sum(integral2, axis=(1, 2))

        return kbar


class SMMModelGriffithFlaw(CrackShapeDependent):
    """
    # For now just use the von Mises effective stress
    vm = np.sqrt((
        (tube.quadrature_results['stress_xx'] - tube.quadrature_results['stress_yy'])**2.0 +
        (tube.quadrature_results['stress_yy'] - tube.quadrature_results['stress_zz'])**2.0 +
        (tube.quadrature_results['stress_zz'] - tube.quadrature_results['stress_xx'])**2.0 +
        6.0 * (tube.quadrature_results['stress_xy']**2.0 +
          tube.quadrature_results['stress_yz']**2.0 +
          tube.quadrature_results['stress_xz']**2.0))/2.0)

    tR = material.time_to_rupture("averageRupture", tube.quadrature_results['temperature'], vm)
    dts = np.diff(tube.times)
    time_dmg = dts[:,np.newaxis,np.newaxis]/tR[1:]

    # Break out to cycle damage
    inds = self.id_cycles(tube, receiver)

    cycle_dmg = np.array([
      np.sum(time_dmg[inds[i]:inds[i+1]], axis = 0) for i in range(receiver.days)])

    return cycle_dmg

  def fatigue_damage(self, tube, material, receiver):
    """

    def calculate_eq_stress(self, mandel_stress, temperatures, material):
        """
        Calculate the equivalent stresses from the normal and shear stresses

        Additional parameters:
        cbar:   Shetty model empirical constant (cbar)
        """

        # Material parameters
        cbar = np.mean(material.c_bar(temperatures), axis=0)

        # Normal stress
        sigma_n = self.calculate_normal_stress(mandel_stress)

        # Shear stress
        tau = self.calculate_shear_stress(mandel_stress)

        # Projected equivalent stress
        return 0.5 * (
            sigma_n + np.sqrt((sigma_n**2) + ((2 * tau / cbar[..., None, None]) ** 2))
        )

    def calculate_kbar(self, temperatures, material, A, dalpha, dbeta):
        """
        Calculate the evaluating normalized crack density coefficient (kbar)

        Parameters:
        temperatures:   element temperatures
        material:       material model object with required data that includes
                        Weibull modulus (mvals), poisson ratio (nu),
                        Shetty model empirical constant (cbar)
        """

        self.temperatures = temperatures
        self.material = material

        # Weibull modulus
        mvals = self.material.modulus(temperatures)

        # Temperature average values
        mavg = np.mean(mvals, axis=0)

        # Material parameters
        cbar = np.mean(material.c_bar(temperatures), axis=0)

        # Evaluating integral for kbar
        integral2 = 2 * (
            (
                (
                    0.5
                    * (
                        ((np.cos(self.A)) ** 2)
                        + np.sqrt(
                            ((np.cos(self.A)) ** 4)
                            + (
                                ((np.sin(2 * self.A)) ** 2)
                                / (cbar[..., None, None] ** 2)
                            )
                        )
                    )
                )
                ** mavg[..., None, None]
            )
            * np.sin(self.A)
            * self.dalpha
            * self.dbeta
        )
        kbar = np.pi / np.sum(integral2, axis=(1, 2))

        return kbar


class SMMModelPennyShapedFlaw(CrackShapeDependent):
    """
    # Identify cycle boundaries
    inds = self.id_cycles(tube, receiver)

    # Run through each cycle and ID max strain range and fatigue damage
    strain_names = ['mechanical_strain_xx', 'mechanical_strain_yy', 'mechanical_strain_zz',
        'mechanical_strain_yz', 'mechanical_strain_xz', 'mechanical_strain_xy']
    strain_factors = [1.0,1.0,1.0,2.0, 2.0, 2.0]

    cycle_dmg =  np.array([self.cycle_fatigue(np.array([ef*tube.quadrature_results[en][
      inds[i]:inds[i+1]] for
      en,ef in zip(strain_names, strain_factors)]),
      tube.quadrature_results['temperature'][inds[i]:inds[i+1]], material)
      for i in range(receiver.days)])

    return cycle_dmg

  def id_cycles(self, tube, receiver):
    """

    def calculate_eq_stress(self, mandel_stress, temperatures, material):
        """
        Calculate the equivalent stresses from the normal and shear stresses

        Additional parameters:
        nu:     Poisson ratio
        cbar:   Shetty model empirical constant (cbar)
        """

        # Material parameters
        nu = np.mean(material.nu(temperatures), axis=0)
        cbar = np.mean(material.c_bar(temperatures), axis=0)

        # Normal stress
        sigma_n = self.calculate_normal_stress(mandel_stress)

        # Shear stress
        tau = self.calculate_shear_stress(mandel_stress)

        # Projected equivalent stress
        return 0.5 * (
            sigma_n
            + np.sqrt(
                (sigma_n**2)
                + ((4 * tau / (cbar[..., None, None] * (2 - nu[..., None, None]))) ** 2)
            )
        )

    def calculate_kbar(self, temperatures, material, A, dalpha, dbeta):
        """
        Calculate the evaluating normalized crack density coefficient (kbar)

        Parameters:
        temperatures:   element temperatures
        material:       material model object with required data that includes
                        Weibull modulus (mvals), poisson ratio (nu),
                        Shetty model empirical constant (cbar)
        """

        self.temperatures = temperatures
        self.material = material

        # Weibull modulus
        mvals = self.material.modulus(temperatures)

        # Temperature average values
        mavg = np.mean(mvals, axis=0)

        # Material parameters
        nu = np.mean(material.nu(temperatures), axis=0)
        cbar = np.mean(material.c_bar(temperatures), axis=0)

        # Evaluating integral for kbar
        integral2 = 2 * (
            (
                (
                    0.5
                    * (
                        ((np.cos(self.A)) ** 2)
                        + np.sqrt(
                            ((np.cos(self.A)) ** 4)
                            + (
                                (4 * ((np.sin(2 * self.A)) ** 2))
                                / (
                                    (cbar[..., None, None] ** 2)
                                    * ((nu[..., None, None] - 2) ** 2)
                                )
                            )
                        )
                    )
                )
                ** mavg[..., None, None]
            )
            * np.sin(self.A)
            * self.dalpha
            * self.dbeta
        )
        kbar = np.pi / np.sum(integral2, axis=(1, 2))

        return kbar


class DamageCalculator:
    """
    tm = np.mod(tube.times, receiver.period)
    inds = list(np.where(tm == 0)[0])
    if len(inds) != (receiver.days + 1):
      raise ValueError("Tube times not compatible with the receiver"
          " number of days and cycle period!")

    return inds

  def cycle_fatigue(self, strains, temperatures, material, nu = 0.5):
    """

    def __init__(self, pset):
        """
        Parameters:
          pset:       damage parameters
        """
        self.extrapolate = pset.get_default("extrapolate", "lump")
        self.order = pset.get_default("order", 1)

    def single_cycles(self, tube, material, receiver):
        """
        Calculate damage for a single tube

        Parameters:
          tube:       fully-populated tube object
          material:   damage material
          receiver:   receiver object (for metadata)
        """
        raise NotImplementedError("Superclass not implemented")

    def determine_life(self, receiver, material, nthreads=1, decorator=lambda x, n: x):
        """
        Determine the life of the receiver by calculating individual
        material point damage and finding the minimum of all points.

        Parameters:
          receiver        fully-solved receiver object
          material        material model ot use

        Additional Parameters:
          nthreads        number of threads
          decorator       progress bar
        """
        # pylint: disable=no-member
        with multiprocess.Pool(nthreads) as p:
            Ns = list(
                decorator(
                    p.imap(
                        lambda x: self.single_cycles(x, material, receiver),
                        receiver.tubes,
                    ),
                    receiver.ntubes,
                )
            )
        N = min(Ns)

        # Results come out as days
        return N

    def make_extrapolate(self, D):
        """
        Return a damage extrapolation function based on self.extrapolate
        giving the damage for the nth cycle

        Parameters:
          D:      raw, per cycle damage
        """
        if self.extrapolate == "lump":
            return lambda N, D=D: N * np.sum(D) / len(D)
        elif self.extrapolate == "last":

            def Dfn(N, D=D):
                N = int(N)
                if N < len(D) - 1:
                    return np.sum(D[:N])
                else:
                    return np.sum(D[:-1]) + D[-1] * N

            return Dfn
        elif self.extrapolate == "poly":
            p = np.polyfit(np.array(list(range(len(D)))) + 1, D, self.order)
            return lambda N, p=p: np.polyval(p, N)
        else:
            raise ValueError(
                "Unknown damage extrapolation approach %s!" % self.extrapolate
            )


class TimeFractionInteractionDamage(DamageCalculator):
    """
    Calculate life using the ASME time-fraction type approach
    """
    pt_temps = np.max(temperatures, axis = 0)

    pt_eranges = np.zeros(pt_temps.shape)

    nt = strains.shape[1]
    for i in range(nt):
      for j in range(nt):
        de = strains[:,j] - strains[:,i]
        eq = np.sqrt(2) / (2*(1+nu)) * np.sqrt(
            (de[0] - de[1])**2 + (de[1]-de[2])**2 + (de[2]-de[0])**2.0
            + 3.0/2.0 * (de[3]**2.0 + de[4]**2.0 + de[5]**2.0)
            )
        pt_eranges = np.maximum(pt_eranges, eq)

    dmg = np.zeros(pt_eranges.shape)
    # pylint: disable=not-an-iterable
    for ind in np.ndindex(*dmg.shape):
      dmg[ind] = 1.0 / material.cycles_to_fail("nominalFatigue", pt_temps[ind], pt_eranges[ind])

    return dmg
