      program test
      implicit double precision (a-h,o-z)
      parameter (mstep=300)
      real*8 xatom(3,400,mstep),force(3,400,mstep)
      real*8 AL(3,3),force_check(3,400,mstep),Ei_check(400,mstep)
      real*8 Ei_DFT(400,mstep),force_DFT(3,400,mstep)
      real*8 Ei_pred(400,mstep),force_pred(3,400,mstep)
      real*8 Etot_DFT(mstep),Etot_pred(mstep)
      integer iatom(400)

      write(6,*) "input nstep (from step1)"
      read(5,*) nstep

      open(10,file="MOVEMENT.test") 
      rewind(10)
      do istep=1,nstep
      read(10,*) natom
      read(10,*) 
      read(10,*) 
      read(10,*) 
      read(10,*) AL(1,1),AL(2,1),AL(3,1)
      read(10,*) AL(1,2),AL(2,2),AL(3,2)
      read(10,*) AL(1,3),AL(2,3),AL(3,3)
      read(10,*) 
      do i=1,natom
      read(10,*) iatom(i),xatom(1,i,istep),xatom(2,i,istep),
     &  xatom(3,i,istep)
      enddo
      read(10,*) ! force
      do i=1,natom
      read(10,*) iatom(i),force_check(1,i,istep),force_check(2,i,istep),
     &     force_check(3,i,istep)
      enddo
      read(10,*) ! velocity
      do i=1,natom
      read(10,*) 
      enddo
      read(10,*) ! Ei
      do i=1,natom
      read(10,*) iatom(i),Ei_check(i,istep)
      enddo
      read(10,*)  ! ----------- line ine
      enddo    ! istep
      close(10)

      open(10,file="energyVV.pred.tot.test")
      rewind(10)
      do istep=1,nstep
      read(10,*) Etot_DFT(istep),Etot_pred(istep)
      enddo
      close(10)

      open(11,file="energyVV.pred.test")
      rewind(11)
      do istep=1,nstep
      do i=1,natom
      read(11,*) it,Ei_DFT(i,istep),Ei_pred(i,istep)
      enddo
      read(11,*)  !--------------
      enddo
      close(11)

      open(12,file="forceVV.pred.test")
      rewind(12)
      do istep=1,nstep
      do i=1,natom
      do ixyz=1,3
      read(12,*) it,force_DFT(ixyz,i,istep),force_pred(ixyz,i,istep)
      enddo
      enddo
      read(12,*)  !--------------
      enddo
      close(12)

      diff_E=0.d0
      diff_F=0.d0
      do istep=1,nstep
      do i=1,natom
      diff_E=diff_E+abs(Ei_DFT(i,istep)-Ei_check(i,istep))
      do k=1,3
      diff_F=diff_F+abs(force_DFT(k,i,istep)-force_check(k,i,istep))
      enddo
      enddo
      enddo
      diff_E=diff_E/(nstep*natom)
      diff_F=diff_F/(nstep*natom)
      write(6,*) "diff_E,diff_F",diff_E,diff_F

      diff_E_DFT=0.d0
      diff_E_pred=0.d0
      do istep=1,nstep
      sum1=0.d0
      sum2=0.d0
      do i=1,natom
      sum1=sum1+Ei_DFT(i,istep)
      sum2=sum2+Ei_pred(i,istep)
      enddo
      diff_E_DFT=diff_E_DTT+abs(Etot_DFT(istep)-sum1)
      diff_E_pred=diff_E_pred+abs(Etot_pred(istep)-sum2)
      enddo
      write(6,*) "Etot-sum_Ei,DFT,pred",diff_E_DFT/nstep,
     &          diff_E_pred/nstep

!ccccccccccccccccccccccccccccccccc

      do istep=2,nstep
      dE_DFT=0.d0
      dE_pred=0.d0
      do i=1,natom
      dx1=xatom(1,i,istep)-xatom(1,i,istep-1)
      if(abs(dx1+1).lt.abs(dx1)) dx1=dx1+1
      if(abs(dx1-1).lt.abs(dx1)) dx1=dx1-1
      dx2=xatom(2,i,istep)-xatom(2,i,istep-1)
      if(abs(dx2+1).lt.abs(dx2)) dx2=dx2+1
      if(abs(dx2-1).lt.abs(dx2)) dx2=dx2-1
      dx3=xatom(3,i,istep)-xatom(3,i,istep-1)
      if(abs(dx3+1).lt.abs(dx3)) dx3=dx3+1
      if(abs(dx3-1).lt.abs(dx3)) dx3=dx3-1

      dx=AL(1,1)*dx1+AL(1,2)*dx2+AL(1,3)*dx3
      dy=AL(2,1)*dx1+AL(2,2)*dx2+AL(2,3)*dx3
      dz=AL(3,1)*dx1+AL(3,2)*dx2+AL(3,3)*dx3

      dE=dx*(force_DFT(1,i,istep)+force_DFT(1,i,istep-1))/2+
     &   dy*(force_DFT(2,i,istep)+force_DFT(2,i,istep-1))/2+
     &   dz*(force_DFT(3,i,istep)+force_DFT(3,i,istep-1))/2
      dE_DFT=dE_DFT+dE
      dE=dx*(force_pred(1,i,istep)+force_pred(1,i,istep-1))/2+
     &   dy*(force_pred(2,i,istep)+force_pred(2,i,istep-1))/2+
     &   dz*(force_pred(3,i,istep)+force_pred(3,i,istep-1))/2
      dE_pred=dE_pred+dE
      enddo
      write(6,*) " DFT,istep,dE,dF*dR",istep,dE_DFT,
     &             Etot_DFT(istep)-Etot_DFT(istep-1)
      write(6,*) "Pred,istep,dE,dF*dR",istep,dE_pred,
     &             Etot_pred(istep)-Etot_pred(istep-1)
      enddo


      stop
      end



      
