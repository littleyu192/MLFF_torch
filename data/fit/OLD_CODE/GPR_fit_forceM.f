       program GPR_fit_forceM
       implicit double precision (a-h,o-z)


       integer lwork
       integer,allocatable,dimension(:) :: iatom,iatom_type,itype_atom
       real*8,allocatable,dimension(:) :: Energy
       real*8,allocatable,dimension(:,:) :: feat,feat2
       real*8,allocatable,dimension(:,:,:) :: feat_type,feat2_type
       integer,allocatable,dimension(:) :: num_neigh,num,num_atomtype
       integer,allocatable,dimension(:,:) :: list_neigh,ind_type

       real*8,allocatable,dimension(:,:,:,:) :: dfeat,dfeat2
       real*8,allocatable,dimension(:,:,:) :: dfeat_type,dfeat2_type

       real*8,allocatable,dimension(:,:) :: AA
       real*8,allocatable,dimension(:) :: BB

       real*8,allocatable,dimension(:,:,:) :: Gfeat_type
       real*8,allocatable,dimension(:,:) :: Gfeat22_type
       real*8,allocatable,dimension(:,:) :: Gfeat_tmp

       real*8,allocatable,dimension(:,:,:) :: AA_type
       real*8,allocatable,dimension(:,:) :: BB_type

       real*8,allocatable,dimension(:,:) :: SS_tmp,SS_tmp2

       integer,allocatable,dimension(:) :: ipiv

       real*8,allocatable,dimension(:,:) :: w_feat
       real*8,allocatable,dimension(:,:,:) :: feat2_ref

       real*8,allocatable,dimension(:,:,:) :: PV
       real*8,allocatable,dimension(:,:) :: feat2_shift,feat2_scale


       real*8,allocatable,dimension(:,:) :: WW,VV,QQ
       real*8,allocatable,dimension(:,:,:,:) :: SS

       real*8,allocatable,dimension(:,:) :: Gfeat2,dGfeat2

       real*8,allocatable,dimension(:,:) :: force

       real*8,allocatable,dimension(:,:) :: xatom
       real*8,allocatable,dimension(:) :: rad_atom,wp_atom
       real*8 AL(3,3),pi,dE,dFx,dFy,dFzi,rad3
       integer power

 
       integer,allocatable,dimension(:) :: num_inv
       integer,allocatable,dimension(:,:) :: index_inv,index_inv2
       integer,allocatable,dimension(:) :: nfeat0,nfeat2,num_ref,
     &    num_refi

       character(len=80) dfeat_n(400)   
       character(len=80),allocatable,dimension (:) :: trainSetFileDir
       character(len=80) trainSetDir
       character(len=90) MOVEMENTDir,dfeatDir,infoDir,trainDataDir,MOVEMENTallDir
       integer sys_num,sys    
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

       pi=4*datan(1.d0)



       open(10,file="fit.input")
       rewind(10)
       read(10,*) ntype,natom,m_neigh,nimage
       allocate(itype_atom(ntype))
       allocate(nfeat0(ntype))
       allocate(nfeat2(ntype))
       allocate(num_ref(ntype))
       allocate(num_refi(ntype))
       allocate(rad_atom(ntype))
       allocate(wp_atom(ntype))

       do i=1,ntype
       read(10,*) itype_atom(i),nfeat0(i),nfeat2(i),num_ref(i),
     &  rad_atom(i),wp_atom(i)
       enddo
       read(10,*) alpha,dist0
       read(10,*) weight_E,weight_E2,weight_F,delta,rad3,power
       close(10)

       dist0=dist0**2

cccccccc Right now, nfeat0,nfeat2,num_ref for different types
cccccccc must be the same. We will change that later, allow them 
cccccccc to be different
       nfeat0m=0
       nfeat2m=0
       num_refm=0
       num_reftot=0
       num_refi(1)=0


       do i=1,ntype
       if(nfeat0(i).gt.nfeat0m) nfeat0m=nfeat0(i)
       if(nfeat2(i).gt.nfeat2m) nfeat2m=nfeat2(i)
       if(num_ref(i).gt.num_refm) num_refm=num_ref(i)
       num_reftot=num_reftot+num_ref(i)
       if(i.gt.1) then
       num_refi(i)=num_refi(i-1)+num_ref(i-1)
       endif
       enddo


       allocate(w_feat(nfeat2m,ntype))
       allocate(feat2_ref(nfeat2m,num_refm,ntype))
       do itype=1,ntype
       open(10,file="weight_feat."//char(itype+48))
       rewind(10)
       do j=1,nfeat2(itype)
       read(10,*) j1,w_feat(j,itype)
       w_feat(j,itype)=w_feat(j,itype)**2
       enddo
       close(10)
       enddo

       do itype=1,ntype
       open(10,file="feat2_ref."//char(itype+48),form="unformatted")
       rewind(10)
       read(10) nfeat2_tmp,num_ref_tmp
       if(nfeat2_tmp.ne.nfeat2(itype)) then
       write(6,*) "nfeat2.not.same,feat2_ref",itype,nfeat2_tmp,
     &     nfeat2(itype)
       stop
       endif
       if(num_ref_tmp.ne.num_ref(itype)) then
       write(6,*) "num_ref.not.same,feat2_ref",itype,num_ref_tmp,
     &    num_ref(itype)
       stop
       endif
       read(10) feat2_ref(1:nfeat2(itype),1:num_ref(itype),itype)
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
     &        nfeat2(itype)
       stop
       endif
       if(nfeat0_tmp.ne.nfeat0(itype)) then
       write(6,*) "nfeat0.not.same,feat2_ref",itype,nfeat0_tmp,
     &         nfeat0(itype)
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
       allocate(AA(num_reftot,num_reftot))
       allocate(BB(num_reftot))
       allocate(AA_type(num_refm,num_refm,ntype))
       allocate(BB_type(num_refm,ntype))

       open(13,file="location")
       rewind(13)
       read(13,*) sys_num  !,trainSetDir
       read(13,'(a80)') trainSetDir
       allocate(trainSetFileDir(sys_num))
       do i=1,sys_num
       read(13,'(a80)') trainSetFileDir(i)    
       enddo
       close(13)

       AA=0.d0
       BB=0.d0

       ave_dist=0.d0
       num_ave_dist=0
       amax_dist=0.d0
       amax_mindist=0.d0

       open(77,file="mindist.dist")
       rewind(77)

       do 900 sys=1,sys_num
       dfeatDir=trim(trainSetFileDir(sys))//"/dfeat.fbin"
       if (sys.ne.1) then    

       deallocate(iatom_type)
       deallocate(iatom)
       deallocate(Energy)
       deallocate(feat)
       deallocate(feat2)
       deallocate(feat_type)
       deallocate(feat2_type)
       deallocate(Gfeat22_type)
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
       deallocate(VV)
       deallocate(SS)

       deallocate(Gfeat_type)
       deallocate(Gfeat_tmp)

       deallocate(SS_tmp)
       deallocate(SS_tmp2)

       deallocate(Gfeat2)
       deallocate(dGfeat2)
       deallocate(WW)
       deallocate(QQ)

       endif
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

       allocate(Gfeat_type(natom,num_refm,ntype))
       allocate(Gfeat22_type(num_refm,ntype))
       allocate(Gfeat_tmp(natom,num_refm))

       allocate(SS_tmp(3*natom,num_reftot))
       allocate(SS_tmp2(3*natom,num_reftot))
       allocate(ipiv(num_reftot))
       allocate(Gfeat2(num_refm,natom))
       allocate(dGfeat2(num_refm,natom))
       allocate(num_inv(natom))
       allocate(index_inv(3*m_neigh,natom))
       allocate(index_inv2(3*m_neigh,natom))
       allocate(force(3,natom))
       allocate(WW(nfeat2m,num_refm))
       allocate(VV(nfeat2m,3*natom))
       allocate(QQ(num_refm,3*natom))
       allocate(SS(natom,3,num_refm,ntype))




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
       read(10) dfeat
       read(10) xatom
       read(10) AL

       
       num=0
       do i=1,natom
       itype=iatom_type(i)
       num(itype)=num(itype)+1
       ind_type(num(itype),itype)=i
       feat_type(:,num(itype),itype)=feat(:,i)
       enddo

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
       if (iatom_type(i) .ne. iatom_type(j)) rad=rad3
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
       dE=dE+0.5*4*w22*(rad/dd)**power*cos(yy)**2
       dEdd=4*w22*(-power*(rad/dd)**power/dd*cos(yy)**2
     &   -(pi/(2*rad))*cos(yy)*sin(yy)*(rad/dd)**power)

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



       do itype=1,ntype
       call dgemm('T','N',nfeat2(itype),num(itype),nfeat0(itype),1.d0,
     & PV(1,1,itype),
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

ccccccccccccccccccccccccccccccccccccccccccc
cccccccccccccccccccccccccccccccccccccc  new
cccccccccccccccccccccccccccccccccccccccccccccccccccccc
ccc  Now, we have yield the new feature: feat2_type(nfeat2,num(itype),itype)
cccc The last feature, nfeat2, =1
cccccccccccccccccccccccccccccccccccccccccccccccccccccc

ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
cccc We calculate the reference here: Gfeat_type(num(itype),num_ref,itype)
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

       do 200 itype=1,ntype

       do i=1,num(itype) 

       amindist=1000000.d0
       ave_dis_each=0.d0
       do k=1,num_ref(itype)-1
       dist=0.d0
       do j=1,nfeat2(itype)-1     
       dist=dist+(feat2_type(j,i,itype)-feat2_ref(j,k,itype))**2*
     &   w_feat(j,itype)
       enddo
       if(dist.lt.amindist) amindist=dist
       ave_dis_each=ave_dis_each+dist

       ave_dist=ave_dist+dist
       num_ave_dist=num_ave_dist+1
       if(dist.gt.amax_dist) amax_dist=dist

ccccccccccccccc  This might be changed later
ccccc The kernel
!       Gfeat_type(i,k,itype)=1/(dist**alpha+dist0**alpha)
       Gfeat_type(i,k,itype)=exp(-(dist/dist0)**alpha)
ccccccccccccccc  This might be changed later
       enddo
       Gfeat_type(i,num_ref(itype),itype)=1   ! important !
       if(amindist.gt.amax_mindist) amax_mindist=amindist
       write(77,*) dsqrt(amindist),dsqrt(ave_dis_each/num_ref(itype))

       enddo
       Gfeat_tmp(:,:)=Gfeat_type(:,:,itype)


       call dgemm('T','N',num_ref(itype),num_ref(itype),num(itype),
     & 1.d0,Gfeat_type(1,1,itype),natom,Gfeat_tmp,natom,
     & 0.d0,AA_type(1,1,itype),num_refm)


       do k=1,num_ref(itype)
       sum=0.d0
       do i=1,num(itype)
       sum=sum+energy(ind_type(i,itype))*Gfeat_type(i,k,itype)
       enddo
       BB_type(k,itype)=sum
       enddo

200    continue
       
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        do itype=1,ntype
        do k=1,num_ref(itype)
        sum=0.d0
        do i=1,num(itype)
        sum=sum+Gfeat_type(i,k,itype)
        enddo
        Gfeat22_type(k,itype)=sum
        enddo
        enddo

        Etot=0.d0
        do i=1,natom
        Etot=Etot+energy(i)
        enddo

ccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
cccc Now, we have finished the energy part. In the following, we will 
cccc include the force part. Which is more complicated. 
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

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
     &  1.d0,PV(1,1,itype),
     & nfeat0m,dfeat_type(1,1,itype),nfeat0m,0.d0,
     & dfeat2_type(1,1,itype),nfeat2m)
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

       do i=1,natom
       itype=iatom_type(i)  ! this is this neighbor's type

       do k=1,num_ref(itype)-1
       dist=0.d0
       do j=1,nfeat2(itype)-1     ! The last feature one is 1. 
       dist=dist+(feat2(j,i)-feat2_ref(j,k,itype))**2*
     &   w_feat(j,itype)
       enddo
!       xx=dist**alpha+dist0**alpha
!       Gfeat2(k,i)=1/xx
!       dGfeat2(k,i)=-1/xx**2*alpha*dist**(alpha-1)   ! derivative
       
       xx=exp(-(dist/dist0)**alpha)
       Gfeat2(k,i)=xx
       dGfeat2(k,i)=-alpha/dist0*(dist/dist0)**(alpha-1)*xx

       enddo
       Gfeat2(num_ref(itype),i)=1   ! important !
       dGfeat2(num_ref(itype),i)=0.d0
       enddo
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
ccccc Now, the most expensive loop

       num_inv=0
       do i=1,natom
       do j=1,num_neigh(i)
       ii=list_neigh(j,i)
       num_inv(ii)=num_inv(ii)+1
       index_inv(num_inv(ii),ii)=i
       index_inv2(num_inv(ii),ii)=j
       enddo
       enddo

ccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

       SS=0d0
       QQ=0.d0
  
cccccccccc A very heavy do loop
       do 100 ii=1,natom
       itype=iatom_type(ii)

        do k=1,num_ref(itype)
        do j=1,nfeat2(itype)
        WW(j,k)=2*(feat2(j,ii)-feat2_ref(j,k,itype))*dGfeat2(k,ii)*
     &   w_feat(j,itype)
        enddo
        enddo

        nmax=num_inv(ii)    
        do jj=1,nmax
        do j=1,nfeat2(itype)
        VV(j,(jj-1)*3+1)=dfeat2(j,index_inv(jj,ii),index_inv2(jj,ii),1)
        VV(j,(jj-1)*3+2)=dfeat2(j,index_inv(jj,ii),index_inv2(jj,ii),2)
        VV(j,(jj-1)*3+3)=dfeat2(j,index_inv(jj,ii),index_inv2(jj,ii),3)
        enddo
        enddo

       call dgemm('T','N',num_ref(itype),3*nmax,nfeat2(itype),1.d0,WW,
     & nfeat2m,VV,nfeat2m,0.d0,QQ,num_refm)
        
        do jj=1,nmax
        do k=1,num_ref(itype)
        SS(index_inv(jj,ii),1,k,itype)=SS(index_inv(jj,ii),1,k,itype)+
     &      QQ(k,(jj-1)*3+1)
        SS(index_inv(jj,ii),2,k,itype)=SS(index_inv(jj,ii),2,k,itype)+
     &      QQ(k,(jj-1)*3+2)
        SS(index_inv(jj,ii),3,k,itype)=SS(index_inv(jj,ii),3,k,itype)+
     &      QQ(k,(jj-1)*3+3)
        enddo
        enddo
100     continue
ccccccccccccccccccccccccccccccccccccccccccc

       SS_tmp=0.d0

       do i=1,natom

       do itype=1,ntype
       do k=1,num_ref(itype)
       SS_tmp((i-1)*3+1,k+num_refi(itype))=SS(i,1,k,itype)
       SS_tmp((i-1)*3+2,k+num_refi(itype))=SS(i,2,k,itype)
       SS_tmp((i-1)*3+3,k+num_refi(itype))=SS(i,3,k,itype)
       enddo
       enddo
       enddo

       SS_tmp2=SS_tmp
       
       call dgemm('T','N',num_reftot,num_reftot,3*natom,
     & weight_F,SS_tmp,3*natom,SS_tmp2,3*natom,1.d0,AA,num_reftot)

ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

       
        do itype=1,ntype
        do k=1,num_ref(itype)
        sum=0.d0
        do i=1,natom
        do ixyz=1,3
        sum=sum+SS(i,ixyz,k,itype)*Force(ixyz,i)
        enddo
        enddo
        BB(k+num_refi(itype))=BB(k+num_refi(itype))+sum*weight_F
        enddo
        enddo


        do itype=1,ntype
        iii=num_refi(itype)
        do k1=1,num_ref(itype)
        do k2=1,num_ref(itype)
        AA(k1+iii,k2+iii)=AA(k1+iii,k2+iii)+
     &              weight_E*AA_type(k1,k2,itype)
        enddo
        enddo
        enddo

        do itype=1,ntype
        iii=num_refi(itype)
        do k=1,num_ref(itype)
        BB(k+iii)=BB(k+iii)+weight_E*BB_type(k,itype)
        enddo
        enddo


cccccccccccccccccccccccccccccccccccccccccc
        do itype1=1,ntype
        iii1=num_refi(itype1)
        do k1=1,num_ref(itype1)
        do itype2=1,ntype
        iii2=num_refi(itype2)
        do k2=1,num_ref(itype2)
        AA(k1+iii1,k2+iii2)=AA(k1+iii1,k2+iii2)+
     &  Gfeat22_type(k1,itype1)*Gfeat22_type(k2,itype2)*weight_E2/natom
        enddo
        enddo
        enddo
        enddo

        do itype=1,ntype
        iii=num_refi(itype)
        do k=1,num_ref(itype)
        BB(k+iii)=BB(k+iii)+Gfeat22_type(k,itype)*Etot*weight_E2/natom
        enddo
        enddo

cccccccccccccccccccccccccccccccccccccccccc



3000   continue


       do k1=1,num_reftot
       AA(k1,k1)=AA(k1,k1)+delta!*nimage
       enddo

       close(10)

900    continue

       close(77)

       ave_dist=dsqrt(ave_dist/num_ave_dist)
       amax_dist=dsqrt(amax_dist)
       amax_mindist=dsqrt(amax_mindist)
       open(22,file="ave_dist")
       rewind(22)
       write(22,*) amax_dist,ave_dist,dsqrt(dist0)
       write(22,*) amax_mindist
       close(22)

cccccccccccccccccccccccccccccccccc
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       call dgesv(num_reftot,1,AA,num_reftot,ipiv,BB,
     &     num_reftot,info)  

!cccccccccccccccccccccccccccccccccccccccccccccccccc
       open(10,file="Gfeat_fit.ntype")
       rewind(10)
       write(10,*) num_reftot
       do i=1,num_reftot
       write(10,*) i, BB(i)
       enddo
       close(10)
       

       stop
       end

