       program linear_clusterM
       implicit double precision (a-h,o-z)


       integer lwork
       integer,allocatable,dimension(:) :: iatom,iatom_type,itype_atom
       real*8,allocatable,dimension(:) :: Energy
       real*8,allocatable,dimension(:,:) :: feat,feat2,feat22_type,
     &    feat22_typeN
       real*8,allocatable,dimension(:,:,:) :: feat_type,feat2_type,
     &    feat2_typeN
       integer,allocatable,dimension(:) :: num_neigh,num,num_atomtype,
     &    numf
       integer,allocatable,dimension(:,:) :: list_neigh,ind_type

       real*8,allocatable,dimension(:,:,:,:) :: dfeat,dfeat2
       real*8,allocatable,dimension(:,:,:,:) :: dfeatN,dfeat2N
       real*8,allocatable,dimension(:,:,:) :: dfeat_type,dfeat2_type

       real*8,allocatable,dimension(:,:) :: AA
       real*8,allocatable,dimension(:) :: BB

       real*8,allocatable,dimension(:,:,:) :: Gfeat_type
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
       real*8 AL(3,3),pi,dE,dFx,dFy,dFz,AL_tmp(3,3)

       real*8,allocatable,dimension(:,:) :: xatom_tmp

 
       integer,allocatable,dimension(:) :: num_inv
       integer,allocatable,dimension(:,:) :: index_inv,index_inv2

       integer,allocatable,dimension(:) :: nfeat0,nfeat2,num_ref,
     &    num_refi,nfeat2i,nfeat2iN
       integer,allocatable,dimension(:) :: nfeat2N

       integer,allocatable,dimension(:) :: clusterNum
       integer clusterNumM
       integer kernel_type
       real*8,allocatable,dimension(:) :: alpha0,k_dist0
       real*8,allocatable,dimension(:,:) :: width
       real*8,allocatable,dimension(:,:,:) ::  feat_cent

       real*8 dsum0(3),d_dd(3)
       real*8 akernel(1000),dkernel(1000,3)
       
       real*8, allocatable, dimension (:,:) :: dfeat_tmp
       integer,allocatable, dimension (:) :: iat_tmp,jneigh_tmp,
     &   ifeat_tmp
       integer num_tmp,jj

       ! character(len=80) dfeat_n(400)
       character(len=80) trainSetFileDir(400)
       character(len=80) trainSetDir
       character(len=90) MOVEMENTDir,dfeatDir,infoDir,trainDataDir, 
     &     MOVEMENTallDir
       integer sys_num,sys
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc



       open(10,file="fit.input")
       rewind(10)
       read(10,*) ntype,natom,m_neigh,nimage0
       allocate(itype_atom(ntype))
       allocate(nfeat0(ntype))
       allocate(nfeat2(ntype))
       allocate(nfeat2N(ntype))
       allocate(num_ref(ntype))
       allocate(num_refi(ntype))
       allocate(nfeat2i(ntype))
       allocate(nfeat2iN(ntype))
       allocate(rad_atom(ntype))
       allocate(wp_atom(ntype))
       do i=1,ntype
       read(10,*) itype_atom(i),nfeat0(i),nfeat2(i),num_ref(i),
     &  rad_atom(i),wp_atom(i)
       enddo
       read(10,*) alpha,dist0
       read(10,*) weight_E,weight_E0,weight_F,delta
       close(10)

       dist0=dist0**2


!ccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        

cccccccc Right now, nfeat0,nfeat2,num_ref for different types
cccccccc must be the same. We will change that later, allow them 
cccccccc to be different
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
 
       allocate(ClusterNum(ntype))
       allocate(alpha0(ntype))
       allocate(k_dist0(ntype))
       ClusterNumM=0
       do itype=1,ntype
       open(13,file='feat_cent.'//char(itype+48))
       rewind(13)
       read(13,*) ClusterNum(itype),alpha0(itype)
       if(clusterNum(itype).gt.clusterNumM)  
     &       clusterNumM=clusterNum(itype) 
       close(13)
       enddo

       allocate(feat_cent(nfeat2m,ClusterNumM,ntype))
       allocate(width(ClusterNumM,ntype))

       nfeat2mN=nfeat2m*ClusterNumM

       nfeat2totN=0
       do itype=1,ntype
       nfeat2N(itype)=nfeat2(itype)*ClusterNum(itype)
       nfeat2totN=nfeat2totN+nfeat2N(itype)
       enddo

       nfeat2iN(1)=0
       do itype=2,ntype
       nfeat2iN(itype)=nfeat2iN(itype-1)+nfeat2N(itype-1)
       enddo

!ccccccccccccccccccccccccccccccccccccccccc
!     right now, nfeat2(itype)=nfeat0, it is the same for different type
!     Originally it is different due to PCA. 

       do itype=1,ntype
       open(13,file='feat_cent.'//char(itype+48))
       rewind(13)
       !TODO:
       read(13,*) ClusterNum(itype),alpha0(itype),k_dist0(itype),
     &   kernel_type
       do ii=1,ClusterNum(itype)
       read(13,*) feat_cent(1:nfeat2(itype),ii,itype)
       enddo
       do ii=1,ClusterNum(itype)
       read(13,*) width(ii,itype)
       enddo
       close(13)
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
     &   nfeat2(itype)
       stop
       endif
       if(nfeat0_tmp.ne.nfeat0(itype)) then
       write(6,*) "nfeat0.not.same,feat2_ref",itype,nfeat0_tmp,
     &   nfeat0(itype)
       stop
       endif
       read(11) PV(1:nfeat0(itype),1:nfeat2(itype),itype)
       read(11) feat2_shift(1:nfeat2(itype),itype)
       read(11) feat2_scale(1:nfeat2(itype),itype)
       close(11)
       enddo
!cccccccccccccccccccccccccccccccccccccccccccccccccc

       allocate(num(ntype))
       allocate(numf(ntype))
       allocate(num_atomtype(ntype))
       allocate(AA(nfeat2totN,nfeat2totN))
       allocate(BB(nfeat2totN))
       allocate(AA_type(nfeat2mN,nfeat2mN,ntype))
       allocate(BB_type(nfeat2mN,ntype))

       ! sys_num=400
       open(13,file="location")
       rewind(13)
       read(13,*) sys_num  !,trainSetDir
       read(13,'(a80)') trainSetDir
       !allocate(trainSetFileDir(sys_num))
       do i=1,sys_num
       read(13,'(a80)') trainSetFileDir(i)    
       enddo
       close(13)
       ! MOVEMENTallDir=trim(trainSetDir)//"/MOVEMENT"
       ! trainDataDir=trim(trainSetDir)//"/trainData.txt"

       AA=0.d0
       BB=0.d0

       do 900 sys=1,sys_num
       ! MOVEMENTDir=trim(trainSetFileDir(sys))//"/MOVEMENT"
       dfeatDir=trim(trainSetFileDir(sys))//"/dfeat.fbin"
       ! infoDir=trim(trainSetFileDir(sys))//"/info.txt"
       
           
       if (sys.ne.1) then    

       deallocate(iatom_type)
       deallocate(iatom)
       deallocate(Energy)
       deallocate(feat)
       deallocate(feat2)
       deallocate(feat_type)
       deallocate(feat2_type)
       deallocate(feat2_typeN)
       deallocate(feat22_type)
       deallocate(feat22_typeN)
       deallocate(num_neigh)
       deallocate(list_neigh)
       deallocate(ind_type)
       deallocate(dfeat)
       deallocate(dfeat_type)
       deallocate(dfeat2_type)
       deallocate(dfeat2)
       deallocate(dfeat2N)
       deallocate(xatom)

       deallocate(ipiv)
       deallocate(num_inv)
       deallocate(index_inv)
       deallocate(index_inv2)
       deallocate(force)
       deallocate(VV)
       deallocate(SS)

       endif

       open(10,file=dfeatDir,action="read",!access="stream",
     &     form="unformatted")
       rewind(10)
       read(10) nimage,natom,nfeat0_tmp,m_neigh

!       nimage=nimage0

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
       allocate(feat2_typeN(nfeat2mN,natom,ntype))
       allocate(feat22_type(nfeat2m,ntype))
       allocate(feat22_typeN(nfeat2mN,ntype))
       allocate(num_neigh(natom))
       allocate(list_neigh(m_neigh,natom))
       allocate(ind_type(natom,ntype))
       allocate(dfeat(nfeat0m,natom,m_neigh,3))
       allocate(dfeat_type(nfeat0m,natom*m_neigh*3,ntype))
       allocate(dfeat2_type(nfeat2m,natom*m_neigh*3,ntype))
       allocate(dfeat2(nfeat2m,natom,m_neigh,3))
       allocate(dfeat2N(nfeat2mN,natom,m_neigh,3))
       allocate(xatom(3,natom))

       allocate(ipiv(nfeat2totN))
       allocate(num_inv(natom))
       allocate(index_inv(3*m_neigh,natom))
       allocate(index_inv2(3*m_neigh,natom))
       allocate(force(3,natom))
       allocate(VV(nfeat2totN,3*natom))
       allocate(SS(nfeat2mN,natom,3,ntype))


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


       num_tmp=0


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
       
       numf=0
       do i=1,natom
       itype=iatom_type(i)
       numf(itype)=numf(itype)+1
       ind_type(numf(itype),itype)=i
       feat_type(:,numf(itype),itype)=feat(:,i)
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
!       write(6,"(2(i4,1x),3(f10.5,1x),2x,f13.6)") i,j,dx1,dx2,dx3,dd
       w22=dsqrt(wp_atom(iatom_type(i))*wp_atom(iatom_type(j)))
       yy=pi*dd/(4*rad)
!       dE=dE+0.5*w22*exp((1-dd/rad)*4.0)*cos(yy)**2
!       dEdd=w22*exp((1-dd/rad)*4.d0)*((-4/rad)*cos(yy)**2
!     &   -(pi/(2*rad))*cos(yy)*sin(yy))


       dE=dE+0.5*4*w22*(rad/dd)**12*cos(yy)**2
       dEdd=4*w22*(-12*(rad/dd)**12/dd*cos(yy)**2
     &   -(pi/(2*rad))*cos(yy)*sin(yy)*(rad/dd)**12)
       dFx=dFx-dEdd*dx/dd       ! note, -sign, because dx=d(j)-x(i)
       dFy=dFy-dEdd*dy/dd
       dFz=dFz-dEdd*dz/dd
       endif
       endif
       enddo
!       write(6,*) "dE,dFx",dE,dFx
       energy(i)=energy(i)-dE
       force(1,i)=force(1,i)-dFx   ! Note, assume force=dE/dx, no minus sign
       force(2,i)=force(2,i)-dFy
       force(3,i)=force(3,i)-dFz
       enddo
ccccccccccccccccccccccccccccccccccccccccccc
cccccccccccccccccccccccccccccccccccccc
ccccc We might want to remove all the PV stuff
cccccc Thus, make the nfeat2 and nfeat0 the same


       do itype=1,ntype
       call dgemm('T','N',nfeat2(itype),numf(itype),nfeat0(itype),
     & 1.d0,PV(1,1,itype),
     & nfeat0m,feat_type(1,1,itype),nfeat0m,0.d0,feat2_type(1,1,itype),
     & nfeat2m)
       enddo

       do itype=1,ntype
       do i=1,numf(itype)
       do j=1,nfeat2(itype)-1
       feat2_type(j,i,itype)=(feat2_type(j,i,itype)-
     &  feat2_shift(j,itype))*feat2_scale(j,itype)
       enddo
       feat2_type(nfeat2,i,itype)=1.d0
       enddo
       enddo

       numf=0
       do i=1,natom
       itype=iatom_type(i)
       numf(itype)=numf(itype)+1
       feat2(:,i)=feat2_type(:,numf(itype),itype)
       enddo

cccccccccccccccccccccccccccccccccccccc  new
        do itype=1,ntype
        do j=1,nfeat2(itype)
        sum=0.d0
        do i=1,numf(itype)
        sum=sum+feat2_type(j,i,itype)
        enddo
        feat22_type(j,itype)=sum
        enddo
        enddo


!cccccccccccccccccccccccccccccccccccccccccccccc
!  get the new feat2_typeN, 
!  The feature number is: nfeat2(itype)*clustrNum(itype)
!ccccccccccccccccccccccccccccccc
        do itype=1,ntype
        do i=1,numf(itype)    ! the atom in the system belong to this itype
          iat=ind_type(i,itype)   ! iat, this atom
          sum0=0.d0
          do jj=1,clusterNum(itype)
          sum=0.d0
          do j=1,nfeat2(itype)
          sum=sum+(feat2_type(j,i,itype)-feat_cent(j,jj,itype))**2* 
     &       w_feat(j,itype)
          enddo
!TODO:
          if (kernel_type.eq.1) then
           akernel(jj)=exp(-(sum/width(jj,itype))**alpha0(itype))
          else if(kernel_type.eq.2) then
        akernel(jj)=1/(sum**alpha0(itype)+k_dist0(itype)**alpha0(itype))
          end if
          sum0=sum0+akernel(jj)
          enddo
          if(sum0.lt.1.E-10) then
            sum0=1.0
          end if
          akernel=akernel/sum0

          

          do jj=1,clusterNum(itype)
          do j=1,nfeat2(itype)
          jjj=(jj-1)*nfeat2(itype)+j
          feat2_typeN(jjj,i,itype)=feat2_type(j,i,itype)*akernel(jj)
          enddo
          enddo
        enddo
        enddo
!cccccccc feat2_typeN is the new feature, feature number nfeat2N!
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        do itype=1,ntype
        do j=1,nfeat2N(itype)
        sum=0.d0
        do i=1,numf(itype)
        sum=sum+feat2_typeN(j,i,itype)
        enddo
        feat22_typeN(j,itype)=sum
        enddo
        enddo
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc


        Etot=0.d0
        do i=1,natom
        Etot=Etot+energy(i)
        enddo

cccccccccccccccccccccccccccccccccccccccccccccccccccccc
ccc  Now, we have yield the new feature: feat2_type(nfeat2,num(itype),itype)
cccc The last feature, nfeat2, =1
cccccccccccccccccccccccccccccccccccccccccccccccccccccc

ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

       do 200 itype=1,ntype

       call dgemm('N','T',nfeat2N(itype),nfeat2N(itype),numf(itype),   
     &  1.d0,feat2_typeN(1,1,itype),nfeat2mN,feat2_typeN(1,1,itype),
     &  nfeat2mN,0.d0,AA_type(1,1,itype),nfeat2mN)
      

       do j=1,nfeat2N(itype)
       sum=0.d0
       do i=1,numf(itype)
       sum=sum+energy(ind_type(i,itype))*feat2_typeN(j,i,itype)
       enddo
       BB_type(j,itype)=sum
       enddo

200    continue
       
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
cccc Now, we have finished the energy part. In the following, we will 
cccc include the force part. Which is more complicated. 
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

cccccccccccccccccccccccccccccccccccccccccccccccccccccccc
ccc dfeat(nfeat0,natom,j_neigh,3): dfeat(j,i,jj,3)= d/dR_i(feat(j,list_neigh(jj,i))
ccccccccccccc
cccccccccccccccccccccccccccccccccccccccccccccccccccccccc
ccccc   We should remove the PCA part

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
!  Now, we need to get dfeat2_N(jjj,i,jj,3)

        do iat=1,natom
        do jn=1,num_neigh(iat)
          jat=list_neigh(jn,iat)  ! jat is the neigh of iat
          jtype=iatom_type(jat)  ! the neighbor atom type, the following is doing jat

ccccccccccccc  note: dfeat2(j,iat,jn,3) is the j-th feature of jat, not iat

          sum0=0.d0
          do jj=1,clusterNum(jtype)
          dd=0.d0
          d_dd(:)=0.d0
          do j=1,nfeat2(jtype)   ! right now, this is the sme as nfeat2(itype)
          dxx=feat2(j,jat)-feat_cent(j,jj,jtype)
          dd=dd+dxx**2*w_feat(j,jtype)
          d_dd(:)=d_dd(:)+2*dxx*dfeat2(j,iat,jn,:)*w_feat(j,jtype)
          enddo
!TODO:
          if (kernel_type.eq.1) then
            akernel(jj)=exp(-(dd/width(jj,jtype))**alpha0(jtype))
            dexp_dd=-akernel(jj)*alpha0(jtype)*dd**(alpha0(jtype)-1) 
     &                            /width(jj,jtype)**alpha0(jtype)
            dkernel(jj,:)=dexp_dd*d_dd(:)
           else if(kernel_type.eq.2) then
        akernel(jj)=1/(dd**alpha0(itype)+k_dist0(itype)**alpha0(itype))
             !       xx=dist**alpha+dist0**alpha
!       Gfeat2(k,i)=1/xx
!       dGfeat2(k,i)=-1/xx**2*alpha*dist**(alpha-1)   ! derivative
            dexp_dd=-akernel(jj)**2*alpha0(jtype)*dd**(alpha0(jtype)-1) 
            dkernel(jj,:)=dexp_dd*d_dd(:)
           end if
          sum0=sum0+akernel(jj)
          enddo

        if (sum0.lt.1.E-10) then
          sum0=1.0
        end if

          dsum0(:)=0.d0
          do jj=1,clusterNum(jtype)
          dsum0(:)=dsum0(:)+dkernel(jj,:)
          enddo

          do jj=1,clusterNum(jtype)
          dkernel(jj,:)=dkernel(jj,:)/sum0-akernel(jj)/sum0**2*dsum0(:)
          akernel(jj)=akernel(jj)/sum0     ! normalize the kernel
          enddo



!cccccccccccccccccccccccccccccccccccccccc
!  The new feat2_N(jjj)=feat2(j)*akernel0(jj)
!  So,  dfeat2)N(jjj)=dfeat2(j)*akernel0(jj)+feat2(j)*dkenerl0(jj)
!  Now, we calculate dkenerl0(jj)
!   dkernel0(jj)= dkernel(jj)/sum0-akernel(jj)/sum0**2*dsum0
!iccccccccccccccccccccccccccccccccccccccccccccccccccc

          do jj=1,clusterNum(jtype) ! neighboring atom cluster
          do j=1,nfeat2(jtype)
          jjj=(jj-1)*nfeat2(jtype)+j
cccccccccccccccccccccccc
cccc do we have feat2 here yet? 
          dfeat2N(jjj,iat,jn,1:3)=dfeat2(j,iat,jn,1:3)*akernel(jj)+ 
     &           feat2(j,jat)*dkernel(jj,1:3)   ! this is for jat's feature

          enddo
          enddo
       enddo  ! jn
       enddo  ! iat
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!  The classification of type might not be correct
!  After, this, in the following, just replace feat2 dfeat2 with feat2N, dfeat2N
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
     
ccccc Now, we have the new features, we need to calculate the distance to reference state

       SS=0.d0

       do i=1,natom
       do jj=1,num_neigh(i)
       jtype=iatom_type(list_neigh(jj,i))  ! this is this neighbor's type
       do j=1,nfeat2N(jtype)

       SS(j,i,1,jtype)=SS(j,i,1,jtype)+dfeat2N(j,i,jj,1)
       SS(j,i,2,jtype)=SS(j,i,2,jtype)+dfeat2N(j,i,jj,2)
       SS(j,i,3,jtype)=SS(j,i,3,jtype)+dfeat2N(j,i,jj,3)
       enddo
       enddo
       enddo

       do i=1,natom
       do itype=1,ntype
       do j=1,nfeat2N(itype)
       VV(j+nfeat2iN(itype),(i-1)*3+1)=SS(j,i,1,itype) 
       VV(j+nfeat2iN(itype),(i-1)*3+2)=SS(j,i,2,itype) 
       VV(j+nfeat2iN(itype),(i-1)*3+3)=SS(j,i,3,itype) 
       enddo
       enddo
       enddo
    

       call dgemm('N','T',nfeat2totN,nfeat2totN,3*natom,weight_F,
     & VV,nfeat2totN,VV,nfeat2totN,1.d0,AA,nfeat2totN)


       do itype=1,ntype
       do j=1,nfeat2N(itype)
       sum=0.d0
       do i=1,natom
       do ixyz=1,3
       sum=sum+force(ixyz,i)*VV(j+nfeat2iN(itype),(i-1)*3+ixyz)
       enddo
       enddo

       BB(j+nfeat2iN(itype))=BB(j+nfeat2iN(itype))+sum*weight_F
       enddo
       enddo


        do itype=1,ntype
        iii=nfeat2iN(itype)
        do k1=1,nfeat2N(itype)
        do k2=1,nfeat2N(itype)
        AA(k1+iii,k2+iii)=AA(k1+iii,k2+iii)+
     &              weight_E*AA_type(k1,k2,itype)
        enddo
        enddo
        enddo

        do itype=1,ntype
        iii=nfeat2iN(itype)
        do k=1,nfeat2N(itype)
        BB(k+iii)=BB(k+iii)+weight_E*BB_type(k,itype)
        enddo
        enddo


        do itype1=1,ntype
        iii1=nfeat2iN(itype1)
        do k1=1,nfeat2N(itype1)
        do itype2=1,ntype
        iii2=nfeat2iN(itype2)
        do k2=1,nfeat2N(itype2)
        AA(k1+iii1,k2+iii2)=AA(k1+iii1,k2+iii2)+
     &  feat22_typeN(k1,itype1)*feat22_typeN(k2,itype2)*weight_E0/natom
        enddo
        enddo
        enddo
        enddo

        do itype=1,ntype
        iii=nfeat2iN(itype)
        do k=1,nfeat2N(itype)
        BB(k+iii)=BB(k+iii)+feat22_typeN(k,itype)*Etot*weight_E0/natom
        enddo
        enddo



3000   continue



       do k1=1,nfeat2totN
       AA(k1,k1)=AA(k1,k1)+delta
       enddo

       close(10)



900    continue


ccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       call dgesv(nfeat2totN,1,AA,nfeat2totN,ipiv,BB,
     &     nfeat2totN,info)  
!cccccccccccccccccccccccccccccccccccccccccccccccccc
       open(10,file="cluster_fitB.ntype")
       rewind(10)
       write(10,*) nfeat2totN
       do i=1,nfeat2totN
       write(10,*) i, BB(i)
       enddo
       close(10)
       

       stop
       end

       
