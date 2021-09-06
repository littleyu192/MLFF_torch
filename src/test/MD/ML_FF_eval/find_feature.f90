      subroutine find_feature(natom,Rc,Rc2,n2b,n3b1,n3b2, &
        num_neigh,list_neigh, &
        dR_neigh,iat_neigh,ntype,grid2,grid31,grid32, &
        feat_all,dfeat_allR,nfeat0,m_neigh)
      implicit none
      integer natom,n2b,n3b1,n3b2,ntype
      integer nfeat0,m_neigh
      real*8 Rc,Rc2
      real*8 dR_neigh(3,m_neigh,ntype,natom)
      real*8 dR_neigh_alltype(3,m_neigh,natom)
      integer iat_neigh(m_neigh,ntype,natom),list_neigh(m_neigh,ntype,natom)
      integer num_neigh(ntype,natom)
      integer num_neigh_alltype(natom)
      integer nperiod(3)
      integer iflag,i,j,num,iat,itype
      integer i1,i2,i3,itype1,itype2,j1,j2,iat1,iat2
      real*8 d,dx1,dx2,dx3,dx,dy,dz,dd
      real*8 grid2(0:n2b+1)
      real*8 grid31(0:n3b1+1),grid32(0:n3b2+1)
      real*8 pi,pi2,x,f1
      integer iflag_grid

      real*8 feat2(n2b,ntype,natom)
      real*8 dfeat2(n2b,ntype,natom,m_neigh,3)
      real*8 feat3(n3b1*n3b1*n3b2,ntype*(ntype+1)/2,natom)
      real*8 dfeat3(n3b1*n3b1*n3b2,ntype*(ntype+1)/2,natom,m_neigh,3)

      real*8 feat3_tmp(2,m_neigh,ntype)
      real*8 dfeat3_tmp(2,m_neigh,ntype,3)
      integer ind_f(2,m_neigh,ntype,natom)
      real*8 f32(2),df32(2,2,3)
      integer inf_f32(2),k,k1,k2,k12,j12,ii_f,jj,jj1,jj2,nneigh,ii
      real*8 y,y2
      integer itype12,ind_f32(2)
      integer ind_all_neigh(m_neigh,ntype,natom),list_neigh_alltype(m_neigh,natom)

      real*8 feat_all(nfeat0,natom),dfeat_allR(nfeat0,natom,m_neigh,3)
      real*8 dfeat_all(nfeat0,natom,m_neigh,3)


      num_neigh_alltype=0
      do iat=1,natom
      num=1
      list_neigh_alltype(1,iat)=iat   ! the first neighbore is itself
      dR_neigh_alltype(:,1,iat)=0.d0

      do  itype=1,ntype
      do   j=1,num_neigh(itype,iat)


      num=num+1
        if(num.gt.m_neigh) then
        write(6,*) "total num_neigh.gt.m_neigh,stop",m_neigh
        stop
        endif
      ind_all_neigh(j,itype,iat)=num
      list_neigh_alltype(num,iat)=list_neigh(j,itype,iat)
      dR_neigh_alltype(:,num,iat)=dR_neigh(:,j,itype,iat)
      enddo
      enddo
      num_neigh_alltype(iat)=num
      enddo

!ccccccccccccccccccccccccccccccccccccccccc

      pi=4*datan(1.d0)
      pi2=2*pi

      feat2=0.d0
      dfeat2=0.d0
      feat3=0.d0
      dfeat3=0.d0


      do 3000 iat=1,natom


      do 1000 itype=1,ntype
      do 1000 j=1,num_neigh(itype,iat)

      jj=ind_all_neigh(j,itype,iat)

      dd=dR_neigh(1,j,itype,iat)**2+dR_neigh(2,j,itype,iat)**2+dR_neigh(3,j,itype,iat)**2
      d=dsqrt(dd)

      do k=1,n2b+1
      if(grid2(k).ge.d) exit
      enddo
      k=k-1

      if(k.gt.n2b)  k=n2b
!  This point will be in the intervalL [grid2(k),grid2(k+1)]
!  It will have feature contribution:
!  feat2(k,itype,iat),feat2(k+1,itype,iat)
      if(k.lt.n2b) then
      x=(d-grid2(k))/(grid2(k+2)-grid2(k))
      y=(x-0.5d0)*pi2
      f1=0.5d0*(cos(y)+1)
      feat2(k+1,itype,iat)=feat2(k+1,itype,iat)+f1
      y2=-pi*sin(y)/(d*(grid2(k+2)-grid2(k)))
      dfeat2(k+1,itype,iat,jj,:)=dfeat2(k+1,itype,iat,jj,:)+y2*dR_neigh(:,j,itype,iat)
      dfeat2(k+1,itype,iat,1,:)=dfeat2(k+1,itype,iat,1,:)-y2*dR_neigh(:,j,itype,iat)
! Note, (k+1,itype) is the feature inde
      endif

      if(k.gt.0) then
      x=(d-grid2(k-1))/(grid2(k+1)-grid2(k-1))
      y=(x-0.5d0)*pi2
      f1=0.5d0*(cos(y)+1)
      feat2(k,itype,iat)=feat2(k,itype,iat)+f1
      y2=-pi*sin(y)/(d*(grid2(k+1)-grid2(k-1)))
      dfeat2(k,itype,iat,jj,:)=dfeat2(k,itype,iat,jj,:)+y2*dR_neigh(:,j,itype,iat)
      dfeat2(k,itype,iat,1,:)=dfeat2(k,itype,iat,1,:)-y2*dR_neigh(:,j,itype,iat)
!cccccccccc jj=1, is itself
      endif
!ccccccccccccccccccccccccccccccccccccccccccccccccccccc


      do k=1,n3b1+1
      if(grid31(k).ge.d) exit
      enddo
      k=k-1

      if(k.gt.n3b1)  k=n3b1

      if(k.lt.n3b1) then
      x=(d-grid31(k))/(grid31(k+2)-grid31(k))
      y=(x-0.5d0)*pi2
      f1=0.5d0*(cos(y)+1)
      feat3_tmp(1,j,itype)=f1
      ind_f(1,j,itype,iat)=k+1

      y2=-pi*sin(y)/(d*(grid31(k+2)-grid31(k)))
      dfeat3_tmp(1,j,itype,:)=y2*dR_neigh(:,j,itype,iat)

      else
        
      feat3_tmp(1,j,itype)=0.d0
      ind_f(1,j,itype,iat)=1    ! a place holder, no effect
      dfeat3_tmp(1,j,itype,:)=0.d0
      endif

      if(k.gt.0) then
      x=(d-grid31(k-1))/(grid31(k+1)-grid31(k-1))
      y=(x-0.5d0)*pi2
      f1=0.5d0*(cos(y)+1)
      feat3_tmp(2,j,itype)=f1
      ind_f(2,j,itype,iat)=k

      y2=-pi*sin(y)/(d*(grid31(k+1)-grid31(k-1)))
      dfeat3_tmp(2,j,itype,:)=y2*dR_neigh(:,j,itype,iat)

      else
      feat3_tmp(2,j,itype)=0.d0
      ind_f(2,j,itype,iat)=1
      dfeat3_tmp(2,j,itype,:)=0.d0
      endif
!cccccccccccc So, one Rij will always have two features k, k+1  (1,2)
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
1000  continue


!   Now, the three body feature
!ccccccccccccccccccccccccccccccccccccc



      do 2000 itype2=1,ntype
      do 2000 itype1=1,itype2

      itype12=itype1+((itype2-1)*itype2)/2


      do 2000 j2=1,num_neigh(itype2,iat)
      do 2000 j1=1,num_neigh(itype1,iat)

!      if(itype1.eq.itype2.and.j1.ge.j2) goto 2000
      if(itype1.eq.itype2.and.j1.eq.j2) goto 2000

      jj1=ind_all_neigh(j1,itype1,iat)
      jj2=ind_all_neigh(j2,itype2,iat)

      dd=(dR_neigh(1,j1,itype1,iat)-dR_neigh(1,j2,itype2,iat))**2+ &
         (dR_neigh(2,j1,itype1,iat)-dR_neigh(2,j2,itype2,iat))**2+ &
         (dR_neigh(3,j1,itype1,iat)-dR_neigh(3,j2,itype2,iat))**2

      d=dsqrt(dd)

!      if(d.gt.Rc2) goto 2000
      if(d.gt.Rc2.or.d.lt.1.D-4) goto 2000

      do k=1,n3b2+1
      if(grid32(k).ge.d) exit
      enddo
      k=k-1

      if(k.gt.n3b2)  k=n3b2


      if(k.lt.n3b2) then
      x=(d-grid32(k))/(grid32(k+2)-grid32(k))
      y=(x-0.5d0)*pi2
      f1=0.5d0*(cos(y)+1)
      f32(1)=f1
      ind_f32(1)=k+1
      y2=-pi*sin(y)/(d*(grid32(k+2)-grid32(k)))
      df32(1,1,:)=y2*(dR_neigh(:,j1,itype1,iat)-dR_neigh(:,j2,itype2,iat))
      df32(1,2,:)=-df32(1,1,:)
      else
      f32(1)=0.d0
      ind_f32(1)=1
      df32(1,:,:)=0.d0
      endif

      if(k.gt.0) then
      x=(d-grid32(k-1))/(grid32(k+1)-grid32(k-1))
      y=(x-0.5d0)*pi2
      f1=0.5d0*(cos(y)+1)
      f32(2)=f1
      ind_f32(2)=k
      y2=-pi*sin(y)/(d*(grid32(k+1)-grid32(k-1)))
      df32(2,1,:)=y2*(dR_neigh(:,j1,itype1,iat)-dR_neigh(:,j2,itype2,iat))
      df32(2,2,:)=-df32(2,1,:)

      else
      f32(2)=0.d0
      ind_f32(2)=1
      df32(2,:,:)=0.d0
      endif


!cccccccccccccccccccccccc
!   Each R has two k features, so for the three R, we have the following
      do i1=1,2
      do i2=1,2
      do j12=1,2
      k1=ind_f(i1,j1,itype1,iat)
      k2=ind_f(i2,j2,itype2,iat)
      k12=ind_f32(j12)

      ii_f=0
      if(itype1.ne.itype2) then
      ii_f=k1+(k2-1)*n3b1+(k12-1)*n3b1**2
      endif
      if(itype1.eq.itype2.and.k1.le.k2) then
      ii_f=k1+((k2-1)*k2)/2+(k12-1)*(n3b1*(n3b1+1))/2
      endif

      if(ii_f.ne.0) then
      feat3(ii_f,itype12,iat)=feat3(ii_f,itype12,iat)+ &
       feat3_tmp(i1,j1,itype1)*feat3_tmp(i2,j2,itype2)*f32(j12)

      dfeat3(ii_f,itype12,iat,jj1,:)=dfeat3(ii_f,itype12,iat,jj1,:)+ &
        dfeat3_tmp(i1,j1,itype1,:)*feat3_tmp(i2,j2,itype2)*f32(j12)+ &
        feat3_tmp(i1,j1,itype1)*feat3_tmp(i2,j2,itype2)*df32(j12,1,:)

      dfeat3(ii_f,itype12,iat,jj2,:)=dfeat3(ii_f,itype12,iat,jj2,:)+ &
        feat3_tmp(i1,j1,itype1)*dfeat3_tmp(i2,j2,itype2,:)*f32(j12)+ &
        feat3_tmp(i1,j1,itype1)*feat3_tmp(i2,j2,itype2)*df32(j12,2,:)

      dfeat3(ii_f,itype12,iat,1,:)=dfeat3(ii_f,itype12,iat,1,:)- &
        dfeat3_tmp(i1,j1,itype1,:)*feat3_tmp(i2,j2,itype2)*f32(j12)- &
        feat3_tmp(i1,j1,itype1)*dfeat3_tmp(i2,j2,itype2,:)*f32(j12) 

!cccc (ii_f,itype12) is the feature index
      endif

      enddo
      enddo
      enddo
2000   continue


3000   continue

!cccccccccccccccccccccccccccccccccccccccccccccccc
!cccccccccccccccccccccccccccccccccccccccccccccccc
!   Now, we collect everything together, collapse the index (k,itype)
!   and feat2,feat3, into a single feature. 

      do 5000 iat=1,natom
      nneigh=num_neigh_alltype(iat)
      num=0
      do itype=1,ntype
      do k=1,n2b
      num=num+1
      feat_all(num,iat)=feat2(k,itype,iat)
      dfeat_all(num,iat,1:nneigh,:)=dfeat2(k,itype,iat,1:nneigh,:)
      enddo
      enddo

      do itype2=1,ntype
      do itype1=1,itype2

      itype12=itype1+((itype2-1)*itype2)/2

      do k1=1,n3b1
      do k2=1,n3b1
      do k12=1,n3b2

      ii_f=0
      if(itype1.ne.itype2) then
      ii_f=k1+(k2-1)*n3b1+(k12-1)*n3b1**2
      endif
      if(itype1.eq.itype2.and.k1.le.k2) then
      ii_f=k1+((k2-1)*k2)/2+(k12-1)*(n3b1*(n3b1+1))/2
      endif

      if(ii_f.gt.0) then
      num=num+1
      feat_all(num,iat)=feat3(ii_f,itype12,iat)
      dfeat_all(num,iat,1:nneigh,:)=dfeat3(ii_f,itype12,iat,1:nneigh,:)
      endif

      enddo
      enddo
      enddo
      enddo
      nfeat0=num
      enddo

5000  continue


!ccccccccccccccccccccccccccccccccccc
!  Now, we have to redefine the dfeat_all in another way. 
!  dfeat_all(nfeat,iat,jneigh,3) means:
!  d_jth_feat_of_iat/d_R(jth_neigh_of_iat)
!  dfeat_allR(nfeat,iat,jneigh,3) means:
!  d_jth_feat_of_jth_neigh/d_R(iat)
!cccccccccccccccccccccccccccccccccccccc

      do iat=1,natom
      do j=1,num_neigh_alltype(iat)
!ccccccccccccccccccc, this include the one which is itself, j=1

      iat2=list_neigh_alltype(j,iat)

      do j2=1,num_neigh_alltype(iat2)
      if(list_neigh_alltype(j2,iat2).eq.iat) then
      dd=(dR_neigh_alltype(1,j,iat)+dR_neigh_alltype(1,j2,iat2))**2+  &
         (dR_neigh_alltype(2,j,iat)+dR_neigh_alltype(2,j2,iat2))**2+  &
         (dR_neigh_alltype(3,j,iat)+dR_neigh_alltype(3,j2,iat2))**2  

      if(dd.lt.1.E-8) then
 
      do ii_f=1,nfeat0
      dfeat_allR(ii_f,iat2,j2,:)=dfeat_all(ii_f,iat,j,:)
      enddo
      endif
      endif

      enddo

      enddo
      enddo
!ccccccccccccccccccccccccccccccccccccc

      return
      end subroutine find_feature



