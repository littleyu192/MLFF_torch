       program linear_forceM
       implicit double precision (a-h,o-z)


       integer lwork
       integer,allocatable,dimension(:) :: iatom,iatom_type,itype_atom
       real*8,allocatable,dimension(:) :: Energy,energy_pred
       real*8,allocatable,dimension(:,:) :: feat,feat2
       real*8,allocatable,dimension(:,:,:) :: feat_type,feat2_type
       integer,allocatable,dimension(:) :: num_neigh,num,num_atomtype
       integer,allocatable,dimension(:,:) :: list_neigh,ind_type

       real*8,allocatable,dimension(:,:,:,:) :: dfeat,dfeat2
       real*8,allocatable,dimension(:,:,:) :: dfeat_type,dfeat2_type

       real*8,allocatable,dimension(:,:) :: AA
       real*8,allocatable,dimension(:) :: BB

       real*8,allocatable,dimension(:,:,:) :: Gfeat_type
       real*8,allocatable,dimension(:,:) :: Gfeat_tmp

       real*8,allocatable,dimension(:,:,:) :: AA_type
       real*8,allocatable,dimension(:,:) :: BB_type,BB_type0

       real*8,allocatable,dimension(:,:) :: SS_tmp,SS_tmp2

       integer,allocatable,dimension(:) :: ipiv

       real*8,allocatable,dimension(:,:) :: w_feat
       real*8,allocatable,dimension(:,:,:) :: feat2_ref

       real*8,allocatable,dimension(:,:,:) :: PV
       real*8,allocatable,dimension(:,:) :: feat2_shift,feat2_scale


       real*8,allocatable,dimension(:,:) :: WW,VV,QQ
       real*8,allocatable,dimension(:,:,:,:) :: SS

       real*8,allocatable,dimension(:,:) :: Gfeat2,dGfeat2

       real*8,allocatable,dimension(:,:) :: force,force_pred

 
       integer,allocatable,dimension(:) :: num_inv
       integer,allocatable,dimension(:,:) :: index_inv,index_inv2
       ! character(len=80) dfeat_n(400)
       character(len=80) trainSetFileDir(400)
       character(len=80) trainSetDir
       character(len=90) MOVEMENTDir,dfeatDir,infoDir,trainDataDir
       integer sys_num,sys
       
       real*8, allocatable, dimension (:,:) :: dfeat_tmp
       integer,allocatable, dimension (:) :: iat_tmp,jneigh_tmp,
     &   ifeat_tmp
       integer num_tmp,jj

       integer,allocatable,dimension(:) :: nfeat0,nfeat2,num_ref,
     &    num_refi,nfeat2i

       real*8,allocatable,dimension(:,:) :: xatom,xatom_tmp
       real*8,allocatable,dimension(:) :: rad_atom,wp_atom
       real*8 AL(3,3),pi,dE,dFx,dFy,dFz,AL_tmp(3,3)


ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       pi=4*datan(1.d0)

       open(10,file="fit.input")
       rewind(10)
       read(10,*) ntype,natom,m_neigh,nimage
!       allocate(itype_atom(ntype))
        allocate(itype_atom(ntype))
        allocate(nfeat0(ntype))
        allocate(nfeat2(ntype))
        allocate(num_ref(ntype))
        allocate(num_refi(ntype))
        allocate(nfeat2i(ntype))
        allocate(rad_atom(ntype))
        allocate(wp_atom(ntype))
        do i=1,ntype
        read(10,*) itype_atom(i),nfeat0(i),nfeat2(i),num_ref(i),
     &   rad_atom(i),wp_atom(i)
        enddo
       read(10,*) alpha,dist0
       read(10,*) weight_E,weight_E0,weight_F,delta
       close(10)

       dist0=dist0**2

        nfeat0m=0
        nfeat2m=0
        num_refm=0
        num_reftot=0
        num_refi(1)=0
        nfeat2tot=0
        nfeat2i=0
        nfeat2i(1)=0
        do i=1,ntype
        if(nfeat0(i).gt.nfeat0m) nfeat0m=nfeat0(i)
        if(nfeat2(i).gt.nfeat2m) nfeat2m=nfeat2(i)
        if(num_ref(i).gt.num_refm) num_refm=num_ref(i)
        num_reftot=num_reftot+num_ref(i)
        nfeat2tot=nfeat2tot+nfeat2(i)
        if(i.gt.1) then
        num_refi(i)=num_refi(i-1)+num_ref(i-1)
        nfeat2i(i)=nfeat2i(i-1)+nfeat2(i-1)
        endif

        enddo



       allocate(BB(nfeat2tot))
       allocate(BB_type(nfeat2m,ntype))
       allocate(BB_type0(nfeat2m,ntype))

       open(12,file="linear_fitB.ntype")
           rewind(12)
           read(12,*) ntmp
         if(ntmp.ne.nfeat2tot) then
        write(6,*) "ntmp.not.right,linear_fitB.ntype",ntmp,nfeat2tot
          stop
         endif
          do i=1,nfeat2tot
         read(12,*) itmp, BB(i)
          enddo
         close(12)
          do itype=1,ntype
          do k=1,nfeat2(itype)
          BB_type0(k,itype)=BB(k+nfeat2i(itype))
          enddo
          enddo


cccccccc Right now, nfeat0,nfeat2,num_ref for different types
cccccccc must be the same. We will change that later, allow them 
cccccccc to be different

       allocate(w_feat(nfeat2m,ntype))
       do itype=1,ntype
       open(10,file="weight_feat."//char(itype+48))
       rewind(10)
       do j=1,nfeat2(itype)
       read(10,*) j1,w_feat(j,itype)
       w_feat(j,itype)=w_feat(j,itype)**2
       enddo
       close(10)
       enddo

ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       allocate(PV(nfeat0m,nfeat2m,ntype))
       allocate(feat2_shift(nfeat2m,ntype))
       allocate(feat2_scale(nfeat2m,ntype))
       do itype=1,ntype
       open(11,file="feat_PV."//char(itype+48),form="unformatted")
       rewind(11)
       read(11) nfeat0_tmp,nfeat2_tmp
       if(nfeat2_tmp.ne.nfeat2(itype)) then
       write(6,*) "nfeat2.not.same,feat2_ref",itype,nfeat2_tmp,
     &     nfeat2(itype)
       stop
       endif
       if(nfeat0_tmp.ne.nfeat0(itype)) then
       write(6,*) "nfeat0.not.same,feat2_ref",itype,nfeat0_tmp,
     &     nfeat0(itype)
       stop
       endif
       read(11) PV(1:nfeat0(itype),1:nfeat2(itype),itype)
       read(11) feat2_shift(1:nfeat2(itype),itype)
       read(11) feat2_scale(1:nfeat2(itype),itype)
       close(11)
       enddo
!cccccccccccccccccccccccccccccccccccccccccccccccccc

       allocate(num(ntype))
       allocate(num_atomtype(ntype))
       allocate(AA(nfeat2tot,nfeat2tot))
       allocate(AA_type(nfeat2m,nfeat2m,ntype))

       open(13,file="location")
       rewind(13)
       read(13,*) sys_num  !,trainSetDir
       read(13,'(a80)') trainSetDir
       ! allocate(trainSetFileDir(sys_num))
       do i=1,sys_num
       read(13,'(a80)') trainSetFileDir(i)    
       enddo
       close(13)

       AA=0.d0

       open(70,file="energyL.pred.tot") 
       rewind(70)
 
       do itype=1,ntype
       open(20+itype,file="energyL.pred."//char(itype+48))
       rewind(20+itype)
       open(40+itype,file="forceL.pred."//char(itype+48))
       rewind(40+ityp)
       enddo



       do 900 sys=1,sys_num
       dfeatDir=trim(trainSetFileDir(sys))//"/dfeat.fbin"       

       open(10,file=dfeatDir,action="read",!access="stream",
     &     form="unformatted")
       rewind(10)
       read(10) nimage,natom,nfeat0_tmp,m_neigh

       if(nfeat0m.ne.nfeat0_tmp) then
       write(6,*) "nfeat0.ne.nfeat0_tmp,stop",nfeat0m,nfeat0_tmp
       stop
       endif

       allocate(iatom_type(natom))
       allocate(iatom(natom))
       allocate(Energy(natom))
       allocate(Energy_pred(natom))
       allocate(feat(nfeat0m,natom))
       allocate(feat2(nfeat2m,natom))
       allocate(feat_type(nfeat0m,natom,ntype))
       allocate(feat2_type(nfeat2m,natom,ntype))
       allocate(num_neigh(natom))
       allocate(list_neigh(m_neigh,natom))
       allocate(ind_type(natom,ntype))
       allocate(dfeat(nfeat0m,natom,m_neigh,3))
       allocate(dfeat_type(nfeat0m,natom*m_neigh*3,ntype))
       allocate(dfeat2_type(nfeat2m,natom*m_neigh*3,ntype))
       allocate(dfeat2(nfeat2m,natom,m_neigh,3))
       allocate(xatom(3,natom))

       allocate(ipiv(nfeat2tot))
       allocate(num_inv(natom))
       allocate(index_inv(3*m_neigh,natom))
       allocate(index_inv2(3*m_neigh,natom))
       allocate(force(3,natom))
       allocate(force_pred(3,natom))
       allocate(VV(nfeat2tot,3*natom))
       allocate(SS(nfeat2m,natom,3,ntype))


       pi=4*datan(1.d0)

       read(10) iatom      ! 1,2,3,...,ntype

       do i=1,natom
        iitype=0
        do itype=1,ntype
        if(itype_atom(itype).eq.iatom(i)) then
        iitype=itype
        endif
        enddo
        if(iitype.eq.0) then
        write(6,*) "this type not found", iatom(i)
        endif
        iatom_type(i)=iitype
      enddo


       num_atomtype=0
       do i=1,natom
       itype=iatom_type(i)
       num_atomtype(itype)=num_atomtype(itype)+1
       enddo



       do 3000 image=1,nimage

       write(6,*) "image=",image

       AA_type=0.d0
       BB_type=0.d0


       read(10) energy
       read(10) force
       read(10) feat
       read(10) num_neigh
       read(10) list_neigh
!TODO:
       ! read(10) dfeat
       read(10) num_tmp
       allocate(dfeat_tmp(3,num_tmp))
       allocate(iat_tmp(num_tmp))
       allocate(jneigh_tmp(num_tmp))
       allocate(ifeat_tmp(num_tmp))
       read(10) iat_tmp
       read(10) jneigh_tmp
       read(10) ifeat_tmp
       read(10) dfeat_tmp
       
       read(10) xatom    ! xatom(3,natom)
       read(10) AL       ! AL(3,3)

       dfeat(:,:,:,:)=0.0
       do jj=1,num_tmp
       dfeat(ifeat_tmp(jj),iat_tmp(jj),jneigh_tmp(jj),:)=dfeat_tmp(:,jj)
       enddo
       deallocate(dfeat_tmp)
       deallocate(iat_tmp)
       deallocate(jneigh_tmp)
       deallocate(ifeat_tmp)
cccccccccccccccccccccccccccccccccccccccccccccccccc


       
       num=0
       do i=1,natom
       itype=iatom_type(i)
       num(itype)=num(itype)+1
       ind_type(num(itype),itype)=i
       feat_type(:,num(itype),itype)=feat(:,i)
       enddo


       do itype=1,ntype
       call dgemm('T','N',nfeat2(itype),num(itype),nfeat0(itype),
     &  1.d0,PV(1,1,itype),
     & nfeat0m,feat_type(1,1,itype),nfeat0m,0.d0,feat2_type(1,1,itype),
     & nfeat2m)
       enddo


       do itype=1,ntype
       do i=1,num(itype)
       do j=1,nfeat2(itype)-1
       feat2_type(j,i,itype)=(feat2_type(j,i,itype)-
     &  feat2_shift(j,itype))*feat2_scale(j,itype)
       enddo
       feat2_type(nfeat2(itype),i,itype)=1.d0
       enddo
       enddo

       num=0
       do i=1,natom
       itype=iatom_type(i)
       num(itype)=num(itype)+1
       feat2(:,i)=feat2_type(:,num(itype),itype)
       enddo
cccccccccccccccccccccccccccccccccccccccccccccccccccccc
ccc  Now, we have yield the new feature: feat2_type(nfeat2,num(itype),itype)
cccc The last feature, nfeat2, =1
cccccccccccccccccccccccccccccccccccccccccccccccccccccc

ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

       do i=1,natom
       itype=iatom_type(i)
       sum=0.d0
       do j=1,nfeat2(itype)
       sum=sum+feat2(j,i)*BB_type0(j,itype)
       enddo
       energy_pred(i)=sum
       enddo


cccccccccccccccccccccccccccccccccccccccccccccccccccccccc
ccc dfeat(nfeat0,natom,j_neigh,3): dfeat(j,i,jj,3)= d/dR_i(feat(j,list_neigh(jj,i))
ccccccccccccc


       num=0
       do i=1,natom
       do jj=1,num_neigh(i)
       itype=iatom_type(list_neigh(jj,i))  ! this is this neighbor's type
       num(itype)=num(itype)+1
       dfeat_type(:,num(itype),itype)=dfeat(:,i,jj,1)
       num(itype)=num(itype)+1
       dfeat_type(:,num(itype),itype)=dfeat(:,i,jj,2)
       num(itype)=num(itype)+1
       dfeat_type(:,num(itype),itype)=dfeat(:,i,jj,3)
       enddo
       enddo
ccccccccc Note: num(itype) is rather large, in the scane of natom*num_neigh

       do itype=1,ntype
       call dgemm('T','N',nfeat2(itype),num(itype),nfeat0(itype),
     & 1.d0,PV(1,1,itype),
     & nfeat0m,dfeat_type(1,1,itype),nfeat0m,0.d0,
     & dfeat2_type(1,1,itype),
     & nfeat2m)
       enddo


       num=0
       do i=1,natom
       do jj=1,num_neigh(i)
       itype=iatom_type(list_neigh(jj,i))  ! this is this neighbor's type
       num(itype)=num(itype)+1
       do j=1,nfeat2(itype)-1
       dfeat2(j,i,jj,1)=dfeat2_type(j,num(itype),itype)*
     &                    feat2_scale(j,itype)
       enddo
       dfeat2(nfeat2(itype),i,jj,1)=0.d0
       num(itype)=num(itype)+1
       do j=1,nfeat2(itype)-1
       dfeat2(j,i,jj,2)=dfeat2_type(j,num(itype),itype)*
     &                    feat2_scale(j,itype)
       enddo
       dfeat2(nfeat2(itype),i,jj,2)=0.d0
       num(itype)=num(itype)+1
       do j=1,nfeat2(itype)-1
       dfeat2(j,i,jj,3)=dfeat2_type(j,num(itype),itype)*
     &                    feat2_scale(j,itype)
       enddo
       dfeat2(nfeat2(itype),i,jj,3)=0.d0
       enddo
       enddo
      
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
ccc  Now, dfeat2 is: 
ccc dfeat2(nfeat2,natom,j_neigh,3): dfeat2(j,i,jj,3)= d/dR_i(feat2(j,list_neigh(jj,i))
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
ccccc Now, we have the new features, we need to calculate the distance to reference state

       SS=0.d0

       do i=1,natom
       do jj=1,num_neigh(i)
       itype=iatom_type(list_neigh(jj,i))  ! this is this neighbor's type
       do j=1,nfeat2(itype)

       SS(j,i,1,itype)=SS(j,i,1,itype)+dfeat2(j,i,jj,1)
       SS(j,i,2,itype)=SS(j,i,2,itype)+dfeat2(j,i,jj,2)
       SS(j,i,3,itype)=SS(j,i,3,itype)+dfeat2(j,i,jj,3)
       enddo
       enddo
       enddo


       do i=1,natom
       do ixyz=1,3
       sum=0.d0
       do itype=1,ntype
       do j=1,nfeat2(itype)
       sum=sum+SS(j,i,ixyz,itype)*BB_type0(j,itype)
       enddo
       enddo
       force_pred(ixyz,i)=sum
       enddo
       enddo

cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

ccccccccccccccccccccccccccccccccccccccccccc
       do i=1,natom
       rad1=rad_atom(iatom_type(i))
       dE=0.d0
       dFx=0.d0
       dFy=0.d0
       dFz=0.d0
       do jj=1,num_neigh(i)
       j=list_neigh(jj,i)
       if(i.ne.j) then
       rad2=rad_atom(iatom_type(j))
       rad=rad1+rad2
       dx1=mod(xatom(1,j)-xatom(1,i)+100.d0,1.d0)
       if(abs(dx1-1).lt.abs(dx1)) dx1=dx1-1
       dx2=mod(xatom(2,j)-xatom(2,i)+100.d0,1.d0)
       if(abs(dx2-1).lt.abs(dx2)) dx2=dx2-1
       dx3=mod(xatom(3,j)-xatom(3,i)+100.d0,1.d0)
       if(abs(dx3-1).lt.abs(dx3)) dx3=dx3-1
       dx=AL(1,1)*dx1+AL(1,2)*dx2+AL(1,3)*dx3
       dy=AL(2,1)*dx1+AL(2,2)*dx2+AL(2,3)*dx3
       dz=AL(3,1)*dx1+AL(3,2)*dx2+AL(3,3)*dx3
       dd=dsqrt(dx**2+dy**2+dz**2)
       if(dd.lt.2*rad) then
       w22=dsqrt(wp_atom(iatom_type(i))*wp_atom(iatom_type(j)))
       yy=pi*dd/(4*rad)
c       dE=dE+0.5*w22*exp((1-dd/rad)*4.0)*cos(yy)**2
c       dEdd=w22*exp((1-dd/rad)*4.d0)*((-4/rad)*cos(yy)**2
c     &   -(pi/(2*rad))*cos(yy)*sin(yy))
       dE=dE+0.5*4*w22*(rad/dd)**12*cos(yy)**2
       dEdd=4*w22*(-12*(rad/dd)**12/dd*cos(yy)**2
     &   -(pi/(2*rad))*cos(yy)*sin(yy)*(rad/dd)**12)

       dFx=dFx-dEdd*dx/dd       ! note, -sign, because dx=d(j)-x(i)
       dFy=dFy-dEdd*dy/dd
       dFz=dFz-dEdd*dz/dd
       endif
       endif
       enddo
       energy(i)=energy(i)-dE
       force(1,i)=force(1,i)-dFx   ! Note, assume force=dE/dx, no minus sign
       force(2,i)=force(2,i)-dFy
       force(3,i)=force(3,i)-dFz
       enddo
ccccccccccccccccccccccccccccccccccccccccccc



cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       Etot=0.d0
       Etot_pred=0.d0
       do i=1,natom
       Etot=Etot+energy(i)
       Etot_pred=Etot_pred+energy_pred(i)
       enddo
       write(70,*) Etot, Etot_pred


        do i=1,natom
        itype=iatom_type(i)
        write(20+itype,*) energy(i),energy_pred(i)
        write(40+itype,*) force(1,i),force_pred(1,i)
        write(40+itype,*) force(2,i),force_pred(2,i)
        write(40+itype,*) force(3,i),force_pred(3,i)
        enddo

3000   continue

       deallocate(iatom_type)
       deallocate(iatom)
       deallocate(Energy)
       deallocate(Energy_pred)
       deallocate(feat)
       deallocate(feat2)
       deallocate(feat_type)
       deallocate(feat2_type)
       deallocate(num_neigh)
       deallocate(list_neigh)
       deallocate(ind_type)
       deallocate(dfeat)
       deallocate(dfeat_type)
       deallocate(dfeat2_type)
       deallocate(dfeat2)
       deallocate(xatom)

       deallocate(ipiv)
       deallocate(num_inv)
       deallocate(index_inv)
       deallocate(index_inv2)
       deallocate(force)
       deallocate(force_pred)
       deallocate(VV)
       deallocate(SS)


900    continue

       close(70)

       stop
       end

       
