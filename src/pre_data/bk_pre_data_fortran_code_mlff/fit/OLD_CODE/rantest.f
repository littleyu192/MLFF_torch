**==rantest.spg  processed by SPAG 4.52O  at 18:54 on 27 Mar 1996
 
      SUBROUTINE RANTEST(Iseed)
c
c     test and initialize the random number generator
c
      IMPLICIT NONE
      INTEGER Iseed, i
      DOUBLE PRECISION RANF,r
 
      CALL RANSET(Iseed)
c      PRINT *, ' ******** test random numbers ***********'
      DO i = 1, 5
	 r=RANF(Iseed)
c         PRINT *, ' i,ranf() ', i, RANF(Iseed)
      END DO
      RETURN
      END
