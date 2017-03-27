/*
  pia.h

  calculate the area of intersection of a pair of polygons

  This code is a derived from aip.c by Norman Hardy, the original
  can be downoaded from

    http://www.cap-lore.com/MathPhys/IP/

  J.J. Green 2010, 2015
*/

#ifndef PIA_H
#define PIA_H

#ifdef __cplusplus
extern "C"
{
#endif

#include <stddef.h>

  typedef struct { float x, y; } point_t;
  extern float pia_area(const point_t*, size_t, const point_t*, size_t);

#ifdef __cplusplus
}
#endif

#endif
