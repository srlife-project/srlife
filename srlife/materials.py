# pylint: disable=dangerous-default-value, unused-import
"""
  This module contains material models containing thermal, fluid, and
  material properties.  These models can be stored to and recalled from
  XML files for archiving.
"""

from collections import ChainMap
import xml.etree.ElementTree as ET

import numpy as np
import scipy.interpolate as inter

from neml import parse, models


class DeformationMaterial:
    """
    Incredibly thin wrapper around a NEML XML file
    """

    def __init__(self, xmlfile, modelname):
        """
        Parameters:
          xmlfile:    file location for the input file
          modelname:  which model to load
        """
        self.xmlfile = xmlfile
        self.modelname = modelname

    def get_neml_model(self):
        """
        Return the actual model for use in a solve
        """
        return parse.parse_xml(self.xmlfile, self.modelname)


class ThermalMaterial:
    """
    Material thermal properties.

    This object needs to provide:
      1) material name
      2) the conductivity, as a function of temperature and its derivative
      3) the diffusivity, as a function of temperature and its derivative
    """

    def get_dict(self):
        """
        Returns the data as a dictionary
        """
        raise NotImplementedError()

    def get_type(self):
        """
        Return the string type for the data
        """
        raise NotImplementedError()

    @classmethod
    def load(cls, fname, modelname):
        """
        Load from a dictionary

        Parameters:
          fname       filename
        """
        tag, typ = find_name(fname, modelname)
        data = load_node(tag)[modelname]

        if typ == "PiecewiseLinearThermalMaterial":
            return PiecewiseLinearThermalMaterial.load(data)
        elif typ == "ConstantThermalMaterial":
            return ConstantThermalMaterial.load(data)
        else:
            raise ValueError("Unknown ThermalMaterial type %s" % typ)

    def save(self, fname, modelname):
        """
        Save the model to an XML file as modelname

        Parameters:
          fname       filename to use
          modelname   (base tag) to use
        """
        root = ET.Element("models")

        save_node(modelname, self.get_dict(), root, attrib={"type": self.get_type()})

        tree = ET.ElementTree(element=root)
        tree.write(fname)


class PiecewiseLinearThermalMaterial(ThermalMaterial):
    """
    Interpolate thermal properties linearly from a table
    """

    def __init__(self, name, temps, cond, diff):
        """
        Properties:
          name:           material name
          temps:          list of temperature points
          cond:           list of conductivity values
          diff:           list of diffusivity values
        """
        if len(temps) != len(cond) or len(temps) != len(diff):
            raise ValueError(
                "The lists of temperatures, conductivity,"
                "and diffusivity values must have equal lengths!"
            )

        self.name = name
        self.temps = np.array(temps)
        self.cond = np.array(cond)
        self.diff = np.array(diff)

        self.fcond, self.dfcond = make_piecewise(self.temps, self.cond)
        self.fdiff, self.dfdiff = make_piecewise(self.temps, self.diff)

    def conductivity(self, T):
        """
        Conductivity as a function of temperature

        Parameters:
          T       temperature
        """
        return self.fcond(T)

    def diffusivity(self, T):
        """
        Diffusivity as a function of temperature

        Parameters:
          T       temperature
        """
        return self.fdiff(T)

    def dconductivity(self, T):
        """
        Derivative of conductivity as a function of temperature

        Parameters:
          T       temperature
        """
        return self.dfcond(T)

    def ddiffusivity(self, T):
        """
        Derivative of diffusivity as a function of temperature

        Parameters:
          T       temperature
        """
        return self.dfdiff(T)

    def get_type(self):
        """
        String type
        """
        return "PiecewiseLinearThermalMaterial"

    def get_dict(self):
        """
        Pickled dictionary
        """
        return {
            "name": self.name,
            "temps": string_array(self.temps),
            "cond": string_array(self.cond),
            "diff": string_array(self.diff),
        }

    @classmethod
    def load(cls, values):
        """
        Load from a dictionary

        Parameters:
          values  dictionary values
        """
        return cls(
            values["name"],
            destring_array(values["temps"]),
            destring_array(values["cond"]),
            destring_array(values["diff"]),
        )


class ConstantThermalMaterial(ThermalMaterial):
    """
    Constant thermal properties
    """

    def __init__(self, name, k, alpha):
        """
        Properties:
          name:           material name
          k:              conductivity
          alpha:          diffusivity
        """
        self.name = name
        self.cond = k
        self.diff = alpha

    def conductivity(self, T):
        """
        Conductivity as a function of temperature

        Parameters:
          T       temperature
        """
        return T * 0.0 + self.cond

    def diffusivity(self, T):
        """
        Diffusivity as a function of temperature

        Parameters:
          T       temperature
        """
        return T * 0.0 + self.diff

    def dconductivity(self, T):
        """
        Derivative of conductivity as a function of temperature

        Parameters:
          T       temperature
        """
        return T * 0.0

    def ddiffusivity(self, T):
        """
        Derivative of diffusivity as a function of temperature

        Parameters:
          T       temperature
        """
        return T * 0.0

    def get_type(self):
        """
        String type
        """
        return "ConstantThermalMaterial"

    def get_dict(self):
        """
        Pickled dictionary
        """
        return {"name": self.name, "k": str(self.cond), "alpha": str(self.diff)}

    @classmethod
    def load(cls, values):
        """
        Load from a dictionary

        Parameters:
          values  dictionary values
        """
        return cls(values["name"], float(values["k"]), float(values["alpha"]))


class FluidMaterial:
    """
    Properties for convective heat transfer.

    This object needs to store:
      1) Fluid name
      2) A map between a ThermalMaterial name and the corresponding
         temperature-dependent film coefficient and its derivative
    """

    def get_dict(self):
        """
        Returns the data as a dictionary
        """
        raise NotImplementedError()

    def get_type(self):
        """
        Return the string type for the data
        """
        raise NotImplementedError()

    @classmethod
    def load(cls, fname, modelname):
        """
        Load a FluidMaterial object from a file

        Parameters:
          fname       file name to load from
        """
        tag, typ = find_name(fname, modelname)
        data = load_node(tag)[modelname]

        if typ == "PiecewiseLinearFluidMaterial":
            return PiecewiseLinearFluidMaterial.load(data)
        elif typ == "ConstantFluidMaterial":
            return ConstantFluidMaterial.load(data)
        else:
            raise ValueError("Unknown FluidMaterial type %s" % typ)

    def save(self, fname, modelname):
        """
        Save the model to an XML file as modelname

        Parameters:
          fname       filename to use
          modelname   (base tag) to use
        """
        root = ET.Element("models")

        save_node(modelname, self.get_dict(), root, attrib={"type": self.get_type()})

        tree = ET.ElementTree(element=root)
        tree.write(fname)


class ConstantFluidMaterial(FluidMaterial):
    """
    Supply a mapping between the material type and a constant
    film coefficient
    """

    def __init__(self, data):
        """
        Dictionary of the form {name: value} mapping
        a material name to the definition of the piecewise linear map

        Parameters:
          data:       the dictionary
        """
        self.data = data

    def get_dict(self):
        return {k: str(v) for k, v in self.data.items()}

    def get_type(self):
        return "ConstantFluidMaterial"

    @classmethod
    def load(cls, values):
        """
        Load from a dictionary

        Parameters:
          values      dictionary data
        """
        data = {k: float(val) for k, val in values.items()}
        return cls(data)

    def coefficient(self, material, T):
        """
        Return the film coefficient for the given material and temperature

        Parameters:
          material:       material name
          T:              temperatures

        """
        if material in self.data:
            return T * 0.0 + self.data[material]
        else:
            return T * 0.0 + self.data["default"]

    # pylint: disable=unused-argument
    def dcoefficient(self, material, T):
        """
        Return the derivative of the film coefficient with respect to
        temperature for the give material and temperature.

        Parameters:
          material:       material name
          T:              temperatures
        """
        return T * 0.0


class PiecewiseLinearFluidMaterial(FluidMaterial):
    """
    Supply a mapping between the material type and a piecewise linear
    interpolate defining the film coefficient as a function of temperature.
    """

    def __init__(self, data):
        """
        Dictionary of the form {name: (temperatures, values)} mapping
        a material name to the definition of the piecewise linear map

        Parameters:
          data:       the dictionary
        """
        self.data = data

        self.fns = {name: make_piecewise(T, v) for name, (T, v) in data.items()}

    def get_dict(self):
        return {
            k: {"temp": string_array(T), "values": string_array(v)}
            for k, (T, v) in self.data.items()
        }

    def get_type(self):
        return "PiecewiseLinearFluidMaterial"

    @classmethod
    def load(cls, values):
        """
        Load from a dictionary

        Parameters:
          values      dictionary data
        """
        data = {
            k: (destring_array(pair["temp"]), destring_array(pair["values"]))
            for k, pair in values.items()
        }
        return cls(data)

    def coefficient(self, material, T):
        """
        Return the film coefficient for the given material and temperature

        Parameters:
          material:       material name
          T:              temperatures

        """
        if material in self.fns.keys():
            return self.fns[material][0](T)
        else:
            return self.fns["default"][0](T)

    def dcoefficient(self, material, T):
        """
        Return the derivative of the film coefficient with respect to
        temperature for the give material and temperature.

        Parameters:
          material:       material name
          T:              temperatures
        """
        if material in self.fns.keys():
            return self.fns[material][1](T)
        else:
            return self.fns["default"][1](T)


class StructuralMaterial:
    """
    Properties for structural metallic material

    Supply
      1) cycles to failure as a function of temperature and strain range
      2) time to rupture as a function of temperaure and stress
      3) checks creep-fatigue interaction diagram
    """

    def __init__(self, data):
        self.data = data

    def cycles_to_fail(self, pname, temp, erange):
        """
        Returns fatigue cycles to failure at a given temperature and strain range

        Parameters:
          pname:       property name ("nominalFatigue")
          erange:      strain range in mm/mm
          temp:        temperature in K
        """
        pdata = self.data[pname]
        T, a, n, cutoff = [], [], [], []

        for i in pdata:
            T.append(destring_array(pdata[i]["T"]))
            a.append(destring_array(pdata[i]["a"]))
            n.append(destring_array(pdata[i]["n"]))
            cutoff.append(destring_array(pdata[i]["cutoff"]))

            if np.array(a).shape != np.array(n).shape:
                raise ValueError("\tThe lists of a and n must have equal lengths!")

        inds = np.array(T).argsort(axis=0)
        T = np.array(T)[inds]
        a = np.array(a)[inds]
        n = np.array(n)[inds]
        cutoff = np.array(cutoff)[inds]

        if temp > max(T):
            raise ValueError(
                "\ttemperature is out of range for cycle to failure determination"
            )

        for i in range(np.size(T, axis=0)):
            if temp <= T[i]:
                polysum = 0.0
                if erange <= cutoff[i]:
                    erange = cutoff[i][0][0]
                for b, m in zip(a[i][0], n[i][0]):
                    polysum += b * np.log10(erange) ** m
                break

        return 10**polysum

    def time_to_rupture(self, pname, temp, stress):
        """
        Returns time to rupture at a given temperature and stress

        Parameters:
          pname:       property name ("averageRupture" or "lowerboundRupture")
          stress:      stress in MPa
          temp:        temperature in K
        """
        pdata = self.data[pname]

        a = destring_array(pdata["a"])
        n = destring_array(pdata["n"])
        C = destring_array(pdata["C"])

        if a.shape != n.shape:
            raise ValueError("The lists of a and n must have equal lengths!")

        if stress.shape != temp.shape:
            raise ValueError("Stress and temperature must have the same shape!")

        zeros = stress == 0.0
        not_zeros = np.logical_not(zeros)

        res = np.zeros(stress.shape)
        for b, m in zip(a, n):
            res[not_zeros] += b * np.log10(stress[not_zeros]) ** m
        res[not_zeros] = 10.0 ** (res[not_zeros] / temp[not_zeros] - C)
        res[zeros] = np.inf

        return res

    def inside_envelope(self, pname, damage_fatigue, damage_creep):
        """
        Returns True if the point lies inside the design envelope and False if not

        Parameters:
          pname:               property name ("cfinteraction")
          damage_fatigue:      fatigue damage fraction
          creep_fatigue:       creep damage fraction
        """

        if damage_fatigue < 0.0 or damage_creep < 0.0:
            raise ValueError("\tout of range: negative damage fraction")

        pdata = destring_array(self.data[pname])

        x_1 = 0.0
        y_1 = 1.0
        x_2 = pdata[0]
        y_2 = pdata[1]
        x_3 = 1.0
        y_3 = 0.0

        if damage_fatigue < x_2:
            return damage_creep <= (
                (y_2 - y_1) / (x_2 - x_1) * (damage_fatigue - x_1) + y_1
            )
        return damage_creep <= (
            (y_3 - y_2) / (x_3 - x_2) * (damage_fatigue - x_2) + y_2
        )

    @classmethod
    def load(cls, fname, model):
        """
        Load a Structural Material object from a file

        Parameters:
          fname:       file name to load from
          material:    model name
        """
        tag = ET.parse(fname).getroot().find(model)
        return cls(load_node(tag)[model])

    def save(self, fname, modelname):
        """
        Save to a particular file under a particular model name
        """
        root = ET.Element("models")
        save_node(modelname, self.data, root)
        tree = ET.ElementTree(element=root)
        tree.write(fname)


def make_piecewise(x, y):
    """
    Make two piecewise interpolation functions: a piecewise linear
    interpolate between x and y and the corresponding derivative.
    """
    ydiff = np.zeros(y.shape)
    ydiff[:-1] = np.diff(y) / np.diff(x)
    ydiff[-1] = ydiff[-2]

    return inter.interp1d(x, y, fill_value=(ydiff[0], ydiff[-1])), inter.interp1d(
        x, ydiff, kind="previous", fill_value=(0, 0)
    )


def find_name(xmlfile, name):
    """
    Find the base tag with name in an XML file

    Parameters:
      xmlfile:        file name
      name:           tag to look for
    """
    root = ET.parse(xmlfile).getroot()
    tag = root.find(name)

    return tag, tag.attrib["type"]


def save_node(name, entry, node, attrib={}):
    """
    Save a dictionary to a particular node

    Parameters:
      name:     name of the new node
      entry:    entry of interest
      node:     ET parent node object

    Additional parameters:
      attribs   node attributes
    """
    nnode = ET.SubElement(node, name, attrib)
    if isinstance(entry, dict):
        for k, v in entry.items():
            save_node(k, v, nnode)
    else:
        nnode.text = entry


def load_node(node):
    """
    The actual function that does the loading by walking the XML file

    Parameters:
      node:      xml node object from ET
    """
    if len(node) > 0:
        return {node.tag: dict(ChainMap(*(load_node(child) for child in node)))}
    else:
        return {node.tag: node.text}


def string_array(array):
    """
    Make a numpy array a space separated string
    """
    return " ".join(map(str, array))


def destring_array(string):
    """
    Make an array from a space separated string
    """
    return np.array(list(map(float, string.split(" "))))


class CeramicMaterial:
    """
    Properties for structural ceramic material

    Supply
      1) Weibull strength as a function of temperature
      2) Weibull modulus as a function of temperature
      3) c_bar mode mixity parameter
      ... expand later for time-dependent properties
    """

    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def load(cls, fname, model):
        """
        Load a Ceramic material from a file

        Parameters:
          fname:       file name to load from
          material:    model name
        """
        tag, typ = find_name(fname, model)

        if typ == "StandardModel":
            return StandardCeramicMaterial.load(tag)
        else:
            raise ValueError("Unknown ceramic model type %s in damage data!" % typ)


class StandardCeramicMaterial:
    """
    Ceramic material where:

    1) Weibull strength depends on temperature
    2) Weibull modulus depends on temperature
    3) Constant c_bar parameter
    4) Constant Poisson's ratio
    5) Fatigue exponent parameter Nv depends on temperature
    6) Faitgue parameter Bv depends on temperature
    """

    def __init__(
        self,
        su_temperatures,
        threshold_v,
        threshold_s,
        s_temperatures,
        strengths_v,
        strengths_s,
        m_temperatures,
        modulus_v,
        modulus_s,
        c_bar,
        nu,
        Nv_temperatures,
        Nvvals,
        Nsvals,
        Bv_temperatures,
        Bvvals,
        Bsvals,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.su_temperatures = su_temperatures
        self.threhold_v = threshold_v
        self.threhold_s = threshold_s
        self.su_v = inter.interp1d(s_temperatures, threshold_v)
        self.su_s = inter.interp1d(s_temperatures, threshold_s)
        self.s_temperatures = s_temperatures
        self.strengths_v = strengths_v
        self.strengths_s = strengths_s
        self.s0_v = inter.interp1d(s_temperatures, strengths_v)
        self.s0_s = inter.interp1d(s_temperatures, strengths_s)
        self.m_temperatures = m_temperatures
        self.mvals_v = modulus_v
        self.mvals_s = modulus_s
        self.m_v = inter.interp1d(m_temperatures, modulus_v)
        self.m_s = inter.interp1d(m_temperatures, modulus_s)
        self.C = c_bar
        self.nu_val = nu
        self.Nv_temperatures = Nv_temperatures
        self.Nvvals = Nvvals
        self.Nsvals = Nsvals
        self.Nv = inter.interp1d(Nv_temperatures, Nvvals)
        self.Ns = inter.interp1d(Nv_temperatures, Nsvals)
        self.Bv_temperatures = Bv_temperatures
        self.Bvvals = Bvvals
        self.Bsvals = Bsvals
        self.Bv = inter.interp1d(Bv_temperatures, Bvvals)
        self.Bs = inter.interp1d(Bv_temperatures, Bsvals)

    def threshold_vol(self, T):
        """
        Weibull threshold parameter for volume flaws as a function of temperature
        """
        return self.su_v(T)
    
    def threshold_surf(self, T):
        """
        Weibull threshold parameter for surface flaws as a function of temperature
        """
        return self.su_s(T)

    def strength_vol(self, T):
        """
        Weibull strength for volume flaws as a function of temperature
        """
        return self.s0_v(T)

    def strength_surf(self, T):
        """
        Weibull strength for surface flaws as a function of temperature
        """
        return self.s0_s(T)

    def modulus_vol(self, T):
        """
        Weibull modulus for volume flaws as a function of temperature
        """
        return self.m_v(T)

    def modulus_surf(self, T):
        """
        Weibull modulus for surface flaws as a function of temperature
        """
        return self.m_s(T)

    def c_bar(self, T):
        """
        Mode mixity parameter as a function of temperature
        """
        if np.isscalar(T):
            return self.C
        else:
            return self.C * np.ones(T.shape)

    def nu(self, T):
        """
        Poisson's ratio as a function of temperature
        """
        if np.isscalar(T):
            return self.nu_val
        else:
            return self.nu_val * np.ones(T.shape)

    def fatigue_Nv(self, T):
        """
        Fatigue exponent parameter for volume flaws as a function of temperature
        """
        return self.Nv(T)

    def fatigue_Ns(self, T):
        """
        Fatigue exponent parameter for surface flaws as a function of temperature
        """
        return self.Ns(T)

    def fatigue_Bv(self, T):
        """
        Fatigue parameter for volume flaws as a function of temperature
        """
        return self.Bv(T)

    def fatigue_Bs(self, T):
        """
        Fatigue parameter for surface flaws as a function of temperature
        """
        return self.Bs(T)

    @classmethod
    def load(cls, node):
        """
        Load a Ceramic material from a file

        Parameters:
          node:    node with model
        """
        threshold_v = node.find("threshold_vol")
        su_temps = threshold_v.find("temperatures")
        su_vals_v = threshold_v.find("values")

        threshold_s = node.find("threshold_surf")
        su_vals_s = threshold_s.find("values")

        strength_v = node.find("strength_vol")
        s_temps = strength_v.find("temperatures")
        svals_v = strength_v.find("values")

        strength_s = node.find("strength_surf")
        svals_s = strength_s.find("values")

        m_v = node.find("modulus_vol")
        m_temps = m_v.find("temperatures")
        mvals_v = m_v.find("values")

        m_s = node.find("modulus_surf")
        mvals_s = m_s.find("values")

        c_bar = node.find("c_bar")
        nu = node.find("nu")

        Nv = node.find("fatigue_Nv")
        Nv_temps = Nv.find("temperatures")
        Nvvals = Nv.find("values")

        Ns = node.find("fatigue_Ns")
        Nsvals = Ns.find("values")

        Bv = node.find("fatigue_Bv")
        Bv_temps = Bv.find("temperatures")
        Bvvals = Bv.find("values")

        Bs = node.find("fatigue_Bs")
        Bsvals = Bs.find("values")

        return StandardCeramicMaterial(
            np.array(list(map(float, su_temps.text.strip().split()))),
            np.array(list(map(float, su_vals_v.text.strip().split()))),
            np.array(list(map(float, su_vals_s.text.strip().split()))),
            np.array(list(map(float, s_temps.text.strip().split()))),
            np.array(list(map(float, svals_v.text.strip().split()))),
            np.array(list(map(float, svals_s.text.strip().split()))),
            np.array(list(map(float, m_temps.text.strip().split()))),
            np.array(list(map(float, mvals_v.text.strip().split()))),
            np.array(list(map(float, mvals_s.text.strip().split()))),
            float(c_bar.text),
            float(nu.text),
            np.array(list(map(float, Nv_temps.text.strip().split()))),
            np.array(list(map(float, Nvvals.text.strip().split()))),
            np.array(list(map(float, Nsvals.text.strip().split()))),
            np.array(list(map(float, Bv_temps.text.strip().split()))),
            np.array(list(map(float, Bvvals.text.strip().split()))),
            np.array(list(map(float, Bsvals.text.strip().split()))),
        )

    def save(self, fname, modelname):
        """
        Save to a particular file under a particular model name
        """
        root = ET.Element("models")

        base = ET.SubElement(root, modelname, {"type": "StandardModel"})

        # Volume flaw properties
        strength_v = ET.SubElement(base, "strength_vol")
        s_temps = ET.SubElement(strength_v, "temperatures")
        s_temps.text = " ".join(map(str, self.s_temperatures))
        strengths_v = ET.SubElement(strength_v, "values")
        strengths_v.text = " ".join(map(str, self.strengths_v))

        # Surface flaw properties
        strength_s = ET.SubElement(base, "strength_surf")
        strengths_s = ET.SubElement(strength_s, "values")
        strengths_s.text = " ".join(map(str, self.strengths_s))

        # Volume flaw properties
        m_v = ET.SubElement(base, "modulus_vol")
        m_temps = ET.SubElement(m_v, "temperatures")
        m_temps.text = " ".join(map(str, self.m_temperatures))
        mvals_v = ET.SubElement(m_v, "values")
        mvals_v.text = " ".join(map(str, self.mvals_v))

        # Surface flaw properties
        m_s = ET.SubElement(base, "modulus_surf")
        mvals_s = ET.SubElement(m_s, "values")
        mvals_s.text = " ".join(map(str, self.mvals_s))

        c_bar = ET.SubElement(base, "c_bar")
        c_bar.text = str(self.C)

        nu = ET.SubElement(base, "nu")
        nu.text = str(self.nu_val)

        # Volume flaw properties
        Nv = ET.SubElement(base, "fatigue_Nv")
        Nvtemps = ET.SubElement(Nv, "temperatures")
        Nvtemps.text = " ".join(map(str, self.Nv_temperatures))
        Nvvals = ET.SubElement(Nv, "values")
        Nvvals.text = " ".join(map(str, self.Nvvals))

        # Surface flaw properties
        Ns = ET.SubElement(base, "fatigue_Ns")
        Nsvals = ET.SubElement(Ns, "values")
        Nsvals.text = " ".join(map(str, self.Nsvals))

        # Volume flaw properties
        Bv = ET.SubElement(base, "fatigue_Bv")
        Bvtemps = ET.SubElement(Bv, "temperatures")
        Bvtemps.text = " ".join(map(str, self.Bv_temperatures))
        Bvvals = ET.SubElement(Bv, "values")
        Bvvals.text = " ".join(map(str, self.Bvvals))

        # Surface flaw properties
        Bs = ET.SubElement(base, "fatigue_Bs")
        Bsvals = ET.SubElement(Bs, "values")
        Bsvals.text = " ".join(map(str, self.Bsvals))

        tree = ET.ElementTree(element=root)
        tree.write(fname)
