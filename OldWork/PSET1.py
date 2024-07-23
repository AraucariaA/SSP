import math

#PS1-13
def HMSconverter(Hours, Minutes, Seconds, switch):
    """
    If the switch is 1 you will get radians,
    If the switch is 0 you will get degrees
    """

    if switch == 0:
        deg1 = 360*Hours/24
        deg2 = 360*Minutes/(24*60)
        deg3 = 360*Seconds/(24*60*60)
        deg = deg1 + deg2 + deg3
        return deg
    
    if switch == 1:
        rad1 = (2*math.pi)*Hours/24
        rad2 = (2*math.pi)*Minutes/(24*60)
        rad3 = (2*math.pi)*Seconds/(24*60*60)
        rad = rad1 + rad2 + rad3
        return rad


#PS1-14-a
def dot(vec1, vec2):
    product = vec1[0]*vec2[0] + vec1[1]*vec2[1] + vec1[2]*vec2[2]
    return product

#PS1-14-b
def cross(vec1, vec2):
    xpart = vec1[1]*vec2[2] - vec1[2]*vec2[1]
    ypart = vec1[2]*vec2[0] - vec1[0]*vec2[2]
    zpart = vec1[0]*vec2[1] - vec1[1]*vec2[0]
    norm = [xpart, ypart, zpart]

    return norm

#PS1-14-c
def tripleProduct(vec1, vec2, vec3):
    vec4 = cross(vec2,vec3)
    result = dot(vec1, vec4)
    return(result)