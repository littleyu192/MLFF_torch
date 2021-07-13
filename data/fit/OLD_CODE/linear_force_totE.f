       program linear_force_totE
       implicit double precision (a-h,o-z)


       integer lwork
       integer,allocatable,dimension(:) :: iatom,iatom_type,itype_atom
       real*8,allocatable,dimension(:) :: Energy
       real*8,allocatable,dimension(:,:) :: feat,feat2,feat22_type
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

 
       integer,allocatable,dimension(:) :: num_inv
       integer,allocatable,dimension(:,:) :: index_inv,index_inv2

       
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc



       open(10,file="fit.input")
       rewind(10)
       read(10,*) ntype,natom,m_neigh,nimage0
       allocate(itype_atom(ntype))
       read(10,*) (itype_atom(i),i=1,ntype)
       read(10,*) nfeat0,nfeat2
       read(10,*) num_ref
       read(10,*) alpha,dist0
       read(10,*) weight_E,weight_F,delta
       close(10)

       dist0=dist0**2

cccccccc Right now, nfeat0,nfeat2,num_ref for different types
cccccccc must be the same. We will change that later, allow them 
cccccccc to be different

       allocate(w_feat(nfeat2,ntype))
       do itype=1,ntype
       open(10,file="weight_feat."//char(itype+48))
       rewind(10)
       do j=1,nfeat2
       read(10,*) j1,w_feat(j,itype)
       w_feat(j,itype)=w_feat(j,itype)**2
       enddo
       close(10)
       enddo

ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       allocate(PV(nfeat0,nfeat2,ntype))
       allocate(feat2_shift(nfeat2,ntype))
       allocate(feat2_scale(nfeat2,ntype))
       do itype=1,ntype
       open(11,file="feat_PV."//char(itype+48),form="unformatted")
       rewind(11)
       read(11) nfeat0_tmp,nfeat2_tmp
       if(nfeat2_tmp.ne.nfeat2) then
       write(6,*) "nfeat2.not.same,feat2_ref",itype,nfeat2_tmp,nfeat2
       stop
       endif
       if(nfeat0_tmp.ne.nfeat0) then
       write(6,*) "nfeat0.not.same,feat2_ref",itype,nfeat0_tmp,nfeat0
       stop
       endif
       read(11) PV(:,:,itype)
       read(11) feat2_shift(:,itype)
       read(11) feat2_scale(:,itype)
       close(11)
       enddo
!cccccccccccccccccccccccccccccccccccccccccccccccccc

       open(10,file="dfeat.fbin",action="read",access="stream",
     &     form="unformatted")
       rewind(10)
       read(10) nimage,natom,nfeat0_tmp,m_neigh

       nimage=nimage0

       if(nfeat0.ne.nfeat0_tmp) then
       write(6,*) "nfeat0.ne.nfeat0_tmp,stop",nfeat0,nfeat0_tmp
       stop
       endif

       allocate(iatom_type(natom))
       allocate(iatom(natom))
       allocate(Energy(natom))
       allocate(feat(nfeat0,natom))
       allocate(feat2(nfeat2,natom))
       allocate(feat_type(nfeat0,natom,ntype))
       allocate(feat2_type(nfeat2,natom,ntype))
       allocate(feat22_type(nfeat2,ntype))
       allocate(num_neigh(natom))
       allocate(list_neigh(m_neigh,natom))
       allocate(ind_type(natom,ntype))
       allocate(dfeat(nfeat0,natom,m_neigh,3))
       allocate(dfeat_type(nfeat0,natom*m_neigh*3,ntype))
       allocate(dfeat2_type(nfeat2,natom*m_neigh*3,ntype))
       allocate(dfeat2(nfeat2,natom,m_neigh,3))
       allocate(num(ntype))
       allocate(num_atomtype(ntype))
       allocate(AA(ntype*nfeat2,ntype*nfeat2))
       allocate(BB(ntype*nfeat2))
       allocate(AA_type(nfeat2,nfeat2,ntype))
       allocate(BB_type(nfeat2,ntype))
       allocate(ipiv(nfeat2*ntype))
       allocate(num_inv(natom))
       allocate(index_inv(3*m_neigh,natom))
       allocate(index_inv2(3*m_neigh,natom))
       allocate(force(3,natom))
       allocate(VV(ntype*nfeat2,3*natom))
       allocate(SS(nfeat2,natom,3,ntype))



       AA=0.d0
       BB=0.d0


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
       read(10) dfeat

       
       num=0
       do i=1,natom
       itype=iatom_type(i)
       num(itype)=num(itype)+1
       ind_type(num(itype),itype)=i
       feat_type(:,num(itype),itype)=feat(:,i)
       enddo


       do itype=1,ntype
       call dgemm('T','N',nfeat2,num(itype),nfeat0,1.d0,PV(1,1,itype),
     & nfeat0,feat_type(1,1,itype),nfeat0,0.d0,feat2_type(1,1,itype),
     & nfeat2)
       enddo


       do itype=1,ntype
       do i=1,num(itype)
       do j=1,nfeat2-1
       feat2_type(j,i,itype)=(feat2_type(j,i,itype)-
     &  feat2_shift(j,itype))*feat2_scale(j,itype)
       enddo
       feat2_type(nfeat2,i,itype)=1.d0
       enddo
       enddo

       num=0
       do i=1,natom
       itype=iatom_type(i)
       num(itype)=num(itype)+1
       feat2(:,i)=feat2_type(:,num(itype),itype)
       enddo

cccccccccccccccccccccccccccccccccccccc  new
  
       do j=1,nfeat2
       do itype=1,ntype
       sum=0.d0
       do i=1,num(itype)
       sum=sum+feat2_type(j,i,itype)
       enddo
       feat22_type(j,itype)=sum
       enddo
       enddo

       Etot=0.d0
       do i=1,natom
       Etot=Etot+energy(i)
       enddo

cccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

cccccccccccccccccccccccccccccccccccccccccccccccccccccc
ccc  Now, we have yield the new feature: feat2_type(nfeat2,num(itype),itype)
cccc The last feature, nfeat2, =1
cccccccccccccccccccccccccccccccccccccccccccccccccccccc

ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

       do 200 itype=1,ntype

       call dgemm('N','T',nfeat2,nfeat2,num(itype),
     & 1.d0,feat2_type(1,1,itype),nfeat2,feat2_type(1,1,itype),nfeat2,
     & 0.d0,AA_type(1,1,itype),nfeat2)

       do j=1,nfeat2
       sum=0.d0
       do i=1,num(itype)
       sum=sum+energy(ind_type(i,itype))*feat2_type(j,i,itype)
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
       call dgemm('T','N',nfeat2,num(itype),nfeat0,1.d0,PV(1,1,itype),
     & nfeat0,dfeat_type(1,1,itype),nfeat0,0.d0,dfeat2_type(1,1,itype),
     & nfeat2)
       enddo


       num=0
       do i=1,natom
       do jj=1,num_neigh(i)
       itype=iatom_type(list_neigh(jj,i))  ! this is this neighbor's type
       num(itype)=num(itype)+1
       do j=1,nfeat2-1
       dfeat2(j,i,jj,1)=dfeat2_type(j,num(itype),itype)*
     &                    feat2_scale(j,itype)
       enddo
       dfeat2(nfeat2,i,jj,1)=0.d0
       num(itype)=num(itype)+1
       do j=1,nfeat2-1
       dfeat2(j,i,jj,2)=dfeat2_type(j,num(itype),itype)*
     &                    feat2_scale(j,itype)
       enddo
       dfeat2(nfeat2,i,jj,2)=0.d0
       num(itype)=num(itype)+1
       do j=1,nfeat2-1
       dfeat2(j,i,jj,3)=dfeat2_type(j,num(itype),itype)*
     &                    feat2_scale(j,itype)
       enddo
       dfeat2(nfeat2,i,jj,3)=0.d0
       enddo
       enddo
      
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
ccc  Now, dfeat2 is: 
ccc dfeat2(nfeat2,natom,j_neigh,3): dfeat2(j,i,jj,3)= d/dR_i(feat2(j,list_neigh(jj,i))
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
ccccc Now, we have the new features, we need to calculate the distance to reference state

       SS=0.d0

       do i=1,natom
       do j=1,nfeat2
       do jj=1,num_neigh(i)
       itype=iatom_type(list_neigh(jj,i))  ! this is this neighbor's type

       SS(j,i,1,itype)=SS(j,i,1,itype)+dfeat2(j,i,jj,1)
       SS(j,i,2,itype)=SS(j,i,2,itype)+dfeat2(j,i,jj,2)
       SS(j,i,3,itype)=SS(j,i,3,itype)+dfeat2(j,i,jj,3)
       enddo
       enddo
       enddo

       do i=1,natom
       do j=1,nfeat2
       do itype=1,ntype
       VV(j+(itype-1)*nfeat2,(i-1)*3+1)=SS(j,i,1,itype) 
       VV(j+(itype-1)*nfeat2,(i-1)*3+2)=SS(j,i,2,itype) 
       VV(j+(itype-1)*nfeat2,(i-1)*3+3)=SS(j,i,3,itype) 
       enddo
       enddo
       enddo
    

       call dgemm('N','T',ntype*nfeat2,ntype*nfeat2,3*natom,weight_F,
     & VV,ntype*nfeat2,VV,ntype*nfeat2,1.d0,AA,ntype*nfeat2)


       do itype=1,ntype
       do j=1,nfeat2

       sum=0.d0
       do i=1,natom
       do ixyz=1,3
       sum=sum+force(ixyz,i)*VV(j+(itype-1)*nfeat2,(i-1)*3+ixyz)
       enddo
       enddo

       BB(j+(itype-1)*nfeat2)=BB(j+(itype-1)*nfeat2)+sum*weight_F
       enddo
       enddo


        do itype=1,ntype
        iii=(itype-1)*nfeat2
        do k1=1,nfeat2
        do k2=1,nfeat2
        AA(k1+iii,k2+iii)=AA(k1+iii,k2+iii)+
     &              weight_E*AA_type(k1,k2,itype)*0.d0
        enddo
        enddo
        enddo

        do itype=1,ntype
        iii=(itype-1)*nfeat2
        do k=1,nfeat2
        BB(k+iii)=BB(k+iii)+weight_E*BB_type(k,itype)*0.d0
        enddo
        enddo

       do itype1=1,ntype
       iii1=(itype1-1)*nfeat2
       do k1=1,nfeat2
       do itype2=1,ntype
       do k2=1,nfeat2
       iii2=(Itype2-1)*nfeat2
       AA(k1+iii1,k2+iii2)=AA(k1+iii1,k2+iii2)+
     &     feat22_type(k1,itype1)*feat22_type(k2,itype2)*weight_E/natom
       enddo
       enddo
       enddo
       enddo

       do itype=1,ntype
       iii=(itype-1)*nfeat2
       do k=1,nfeat2
       BB(k+iii)=BB(k+iii)+feat22_type(k,itype)*Etot*weight_E/natom
       enddo
       enddo

3000   continue



       do k1=1,ntype*nfeat2
       AA(k1,k1)=AA(k1,k1)+delta*nimage
       enddo

ccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       call dgesv(ntype*nfeat2,1,AA,ntype*nfeat2,ipiv,BB,
     &     ntype*nfeat2,info)  
!cccccccccccccccccccccccccccccccccccccccccccccccccc
       open(10,file="linear_fitB.ntype")
       rewind(10)
       write(10,*) ntype*nfeat2
       do i=1,nfeat2*ntype
       write(10,*) i, BB(i)
       enddo
       close(10)
       

       stop
       end

       
