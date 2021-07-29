       program feat_PCA_metric2
       implicit double precision (a-h,o-z)
       integer(4) :: i
       real*8,allocatable,dimension(:,:,:) :: feat_case
       real*8,allocatable,dimension(:,:) :: feat2_case
       real*8,allocatable,dimension(:,:) :: feat2_case_tmp
       real*8,allocatable,dimension(:,:) :: Ei_case
       real*8,allocatable,dimension(:) :: feat_tmp
       real*8,allocatable,dimension(:,:) :: S
       real*8,allocatable,dimension(:,:) :: PV
       real*8,allocatable,dimension(:) :: EW,work,BB
       real*8,allocatable,dimension(:) :: BB2_ave
       real*8,allocatable,dimension(:,:) :: BB_store
       real*8,allocatable,dimension(:) :: weight_case
       real*8,allocatable,dimension(:) :: E_fit
       real*8,allocatable,dimension(:) :: feat2_shift,feat2_scale
       integer,allocatable,dimension(:) :: ipiv
       integer iatom_type(10),num_case(10)
       real*8 Ei_tmp
       integer itmp,lwork
       character(len=80),allocatable,dimension (:) :: trainSetFileDir
       character(len=80) trainSetDir,BadImageDir
       character(len=90) MOVEMENTDir,dfeatDir,infoDir,trainDataDir,MOVEMENTallDir
       integer sys_num,sys
       
       open(10,file="fit.input") 
       rewind(10)
       read(10,*) ntype,natom,maxNb,nimage
       do i=1,ntype
          read(10,*)iatom_type(i),nfeat,nfeat2
       end do
       close(10)

       open(13,file="location")
       rewind(13)
       read(13,*) sys_num  !,trainSetDir
       read(13,'(a80)') trainSetDir
       close(13)
       ! MOVEMENTallDir=trim(trainSetDir)//"/MOVEMENTall"
       trainDataDir=trim(trainSetDir)//"/trainData.txt"

       ncase=natom*nimage

       ncase=1000000

       num_case=0

       allocate(feat_case(nfeat,ncase,ntype))
       allocate(Ei_case(ncase,ntype))
       allocate(feat_tmp(nfeat))
       

       open(10,file=trainDataDir)
       rewind(10)
       do i=1,ncase

       read(10,*,IOSTAT=ierr) i2,atom,Ei_tmp,(feat_tmp(j),j=1,nfeat)
       if(ierr.ne.0) then
       ncase=i-1
       goto 700
       endif

!       read(10,,*,iostat=istat)i2,atom,Ei_tmp,(feat_tmp(j),j=1,nfeat)
!              if(istat/=0) then
!              exit
!              end if
       itmp=atom+0.0001
       ii=0
       do itype=1,ntype
       if(itmp.eq.iatom_type(itype)) then
       ii=itype
       endif
       enddo
       if(ii.eq.0) then
       write(6,*) "ii=0,stop"
       stop
       endif

       num_case(ii)=num_case(ii)+1
       Ei_case(num_case(ii),ii)=Ei_tmp
       do j=1,nfeat
       feat_case(j,num_case(ii),ii)=feat_tmp(j)
       enddo
       
       enddo
700    continue
       write(6,*) "total case=",ncase
  
       close(10)


       write(6,*) "num_case", (num_case(i),i=1,ntype)

       write(6,*) "ntype=",ntype
      
       write(*,*)(iatom_type(i),i=1,ntype)
 
       write(6,*) "input the itype you like to do"
       read(5,*) itype

!TODO: with PCA:*****************************************
!        allocate(S(nfeat,nfeat))

!        S=0.d0
!        call dgemm('N','T',nfeat,nfeat,num_case(itype),1.d0,
!      &  feat_case(1,1,itype),
!      & nfeat,feat_case(1,1,itype),nfeat,0.d0,S,nfeat)

!        write(6,*) "test1"

!        lwork=10*nfeat
!        allocate(work(lwork))
!        allocate(EW(nfeat))
!        call dsyev('V','U',nfeat,S,nfeat,EW,work,lwork,info)

!        write(6,*) "test2"

!        open(10,file="PCA_eigen_feat."//char(itype+48))
!        rewind(10)
!        do k=1,nfeat
!        write(10,*) i,EW(nfeat-k+1)
!        enddo
!        close(10)

!        write(6,*) "old number of feature=",nfeat
!        write(6,*) "input the number of new features from CPA"
!        read(5,*) nfeat2
! !       nfeat2=192

       ! allocate(PV(nfeat,nfeat2))

       ! do k=1,nfeat2
       ! scale=1/dsqrt(abs(EW(nfeat-k+1)))
       ! do j=1,nfeat
       ! PV(j,k)=S(j,nfeat-k+1)*scale
       ! enddo
       ! enddo

!*********************************************************
!TODO: without PCA
       allocate(PV(nfeat,nfeat2))

       do k=1,nfeat2
       do j=1,nfeat
       if (j.eq.k) then 
       PV(j,k)=1
       else 
       PV(j,k)=0
       endif
       enddo
       enddo



!  PV(j,k) is the vector for principle component k

       allocate(feat2_case(nfeat2,num_case(itype)))

       write(6,*) "test3"
       call dgemm('T','N',nfeat2,num_case(itype),nfeat,1.d0,PV,
     & nfeat,feat_case(1,1,itype),nfeat,0.d0,feat2_case,nfeat2)

! change the last feature just equal one (as a constant)
! feat2_case(ifeat,icase) is the new features
! The nfeat2 feature is replaced by 1. 

       open(10,file="feat_new_stored0."//char(itype+48),
     &     form="unformatted")
       rewind(10) 
       write(10) num_case(itype),nfeat2
       do ii=1,num_case(itype)
       write(10) ii,Ei_case(ii,itype),feat2_case(:,ii)
       enddo
       close(10)

       feat2_case(nfeat2,:)=1.d0
       allocate(feat2_scale(nfeat2))
       allocate(feat2_shift(nfeat2))

       do j=1,nfeat2-1
       sum=0.d0
       do i=1,num_case(itype)
       sum=sum+feat2_case(j,i)
       enddo
       sum_ave=sum/num_case(itype)

       feat2_shift(j)=sum_ave

       sum=0.d0
       do i=1,num_case(itype)
       feat2_case(j,i)=feat2_case(j,i)-sum_ave
       sum=sum+feat2_case(j,i)**2
       enddo
       sum=sum/num_case(itype)
       if (abs(sum).lt.1.E-10) then
              sum=1
       endif
       sum=1/dsqrt(sum)

       feat2_scale(j)=sum

       do i=1,num_case(itype)
       feat2_case(j,i)=sum*feat2_case(j,i)
       enddo
       enddo

       feat2_shift(nfeat2)=0.d0
       feat2_scale(nfeat2)=1.d0
!  The deature is then shifted and normalized, 
       open(10,file="feat_PV."//char(itype+48),form="unformatted")
       rewind(10)
       write(10) nfeat,nfeat2
       write(10) PV
       write(10) feat2_shift
       write(10) feat2_scale
       close(10)
       ! write(6,*) feat2_case
       open(10,file="feat_shift."//char(itype+48))
       rewind(10)
       do i=1,nfeat2
       write(10,*) feat2_shift(i),feat2_scale(i)
       enddo
       close(10)

       write(6,*) "test4"
        write(6,*) "num_case,itype",num_case(itype),itype
       open(10,file="feat_new_stored."//char(itype+48),
     &    form="unformatted")
       rewind(10) 
       write(10) num_case(itype),nfeat2
       do ii=1,num_case(itype)
       write(10) ii,Ei_case(ii,itype),feat2_case(:,ii)
       enddo
       close(10)

!  In above, finish the new feature. We could have stopped here. 
!------------------------------------------------------------------
!------------------------------------------------------------------
!------------------------------------------------------------------
!------------------------------------------------------------------

!    In the following, we will do a linear fitting E= \sum_i W(i) feat2(i)
!    We will do several times, so have a average for W(i)^2. 
!    The everage W(i) will be used as a metrix to measure the distrance
!    Between two points. 


!        deallocate(S)
       allocate(S(nfeat2,nfeat2))
       allocate(BB(nfeat2))
       allocate(BB2_ave(nfeat2))
       allocate(ipiv(nfeat2))
       allocate(E_fit(num_case(itype)))
       allocate(BB_store(nfeat2,100))
       allocate(weight_case(num_case(itype)))

       allocate(feat2_case_tmp(nfeat2,num_case(itype)))

       iseed=-19287
       BB2_ave=0.d0

       write(6,*) "input iseed (negative)"
       read(5,*) iseed


cccc average over different situations
cccc try smooth out the zeros in BB
       do 1000 iii=1,100

       do ii=1,num_case(itype)
       ran=ran1(iseed)
       if(ran.gt.0.5) then
       weight_case(ii)=1.d0       ! random selection of the cases
       else
       weight_case(ii)=0.d0
       endif

       do j=1,nfeat2     
       feat2_case_tmp(j,ii)=feat2_case(j,ii)*weight_case(ii)
       enddo
       enddo
      


       S=0.d0
c       call dgemm('N','T',nfeat2,nfeat2,num_case(itype),1.d0,feat2_case,
c     & nfeat2,feat2_case,nfeat2,0.d0,S,nfeat2)
       call dgemm('N','T',nfeat2,nfeat2,num_case(itype),1.d0,
     & feat2_case_tmp,
     & nfeat2,feat2_case_tmp,nfeat2,0.d0,S,nfeat2)

       sum=0.d0
       do j1=1,nfeat2
       do j2=1,nfeat2
       sum=sum+abs(S(j1,j2))
       enddo
       enddo
       delta=sum/nfeat2**2
       delta=delta*0.0001
       
       do j=1,nfeat2
       S(j,j)=S(j,j)+delta
       enddo

       do j=1,nfeat2
       sum=0.d0
       do i=1,num_case(itype)
c       sum=sum+Ei_case(i,itype)*feat2_case(j,i)
       sum=sum+Ei_case(i,itype)*feat2_case_tmp(j,i)*weight_case(i)
       enddo
       BB(j)=sum
       enddo


       call dgesv(nfeat2,1,S,nfeat2,ipiv,BB,nfeat2,info)  

!    BB is the linear eight: Ei_case(icase)= \sum_i BB(i)* feat2_case(i,icase)

       do j=1,nfeat2
       BB2_ave(j)=BB2_ave(j)+BB(j)**2
       BB_store(j,iii)=BB(j)
       enddo

1000    continue

       BB2_ave=dsqrt(BB2_ave/100)
       open(10,file="weight_feat."//char(itype+48))
       rewind(10) 

       sum1=0.d0
       do j=1,nfeat2-1
       sum1=sum1+BB2_ave(j)
       enddo
       sum1=sum1/(nfeat2-1)

       do j=1,nfeat2
       write(10,"(i5,1x,1(E15.7,1x))") j,BB2_ave(j)
ccccc do not use this
c       write(10,"(i5,1x,1(E15.7,1x))") j,sum1
       enddo
       close(10)

       ampl=0.d0
       diff1=0.d0
       diff2=0.d0

       do j=1,nfeat2-2

       sum0=0.d0
       sum1=0.d0
       do iii=1,100
       sum0=sum0+BB_store(j,iii)
       sum1=sum1+BB_store(j,iii)**2
       enddo
       sum0=sum0/100
       sum1=sqrt(sum1/100)
       do iii=1,100
       diff1=diff1+(sum0-BB_store(j,iii))**2
       diff2=diff2+(sum1-abs(BB_store(j,iii)))**2
       ampl=ampl+abs(BB_store(j,iii))**2
       enddo
       enddo

       diff1=dsqrt(diff1/100/nfeat2)
       diff2=dsqrt(diff2/100/nfeat2)
       ampl=dsqrt(ampl/100/nfeat2)

       write(6,*) "ampl,diff1,diff2",ampl,diff1,diff2

       stop
!cccccccccccccccccccccccccccccccccccccccccccccccccc
       
       
       
       do i=1,num_case(itype)
       sum=0.d0
       do j=1,nfeat2
       sum=sum+BB(j)*feat2_case(j,i)
       enddo
       E_fit(i)=sum
       enddo

       open(10,file="W_fit."//char(itype+48))
       rewind(10) 
       do j=1,nfeat2
       write(10,"(i5,1x,1(E15.7,1x))") j,BB(j)
       enddo
       close(10)

       write(6,*) "test5"
       open(10,file="E_fit."//char(itype+48))
       rewind(10) 
       do i=1,num_case(itype)
       write(10,"(i5,1x,2(E14.7,1x))") ii,Ei_case(i,itype),E_fit(i)
       enddo
       close(10)


       stop
       end

       
