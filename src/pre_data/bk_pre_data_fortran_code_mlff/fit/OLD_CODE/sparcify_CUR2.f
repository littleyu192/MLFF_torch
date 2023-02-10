       program sparcification_CUR
!  This program select a subset of the point from the original sample point
!  To be used as the reference point for Gaussian Process regression
       implicit double precision (a-h,o-z)
       real*8,allocatable,dimension(:) :: weight_case
       real*8,allocatable,dimension(:) :: w_feat
       real*8,allocatable,dimension(:) :: dist_case
       real*8,allocatable,dimension(:) :: feat2_tmp
 
       real*8,allocatable,dimension(:,:) :: dist,dist2
       integer,allocatable,dimension(:,:) :: ind,ind2
       integer,allocatable,dimension(:) :: num_ind,num_ind2
       integer,allocatable,dimension(:) :: irank
       integer,allocatable,dimension(:) :: iflag,ind_select
       integer,allocatable,dimension(:) :: ind_orig_case
       real*8,allocatable,dimension(:) :: weight_orig_case
      

       real*8,allocatable,dimension(:) :: Ei_case
       real*8,allocatable,dimension(:,:) :: feat2_case,feat2_ref
       integer(4) iatom_type(10),i


       open(10,file="fit.input")
       rewind(10)
       read(10,*) ntype,natom,maxNb,nimage
       do i=1,ntype
           read(10,*) iatom_type(i),nfeat
       end do
       close(10)


       write(6,*) "ntype=",ntype
       write(6,*) "input itype"
       write(*,*)(iatom_type(i),i=1,ntype)
       read(5,*) itype



       open(10,file="feat_new_stored0."//char(itype+48),
     &       form="unformatted")
       rewind(10) 
       read(10) num_case0,nfeat2
       write(6,*) "num_case0,nfeat2", num_case0,nfeat2
       num_case = num_case0

       
       allocate(Ei_case(num_case))
       allocate(weight_case(num_case))
       allocate(feat2_case(nfeat2,num_case))
       allocate(feat2_tmp(nfeat2))
       ii=0
       do ii0=1,num_case
       read(10) jj1,Ei_tmp,feat2_tmp
       ii=ii+1
       Ei_case(ii)=Ei_tmp
       feat2_case(:,ii)=feat2_tmp(:)
       enddo
       close(10)

       
       do ii=1,num_case
       sum=0.d0
       do j=1,nfeat2
       sum=sum+feat2_case(j,ii)**2
       enddo
       weight_case(ii)=sum
       enddo

       open(12,file="weight_case."//char(itype+48))
       rewind(12)
       do ii=1,num_case
       write(12,*) ii,weight_case(ii)
       enddo
       close(12)

       write(6,*) "input wanted num_select"
       read(5,*) num_select
       write(6,*) "input iseed random (negative)"
       read(5,*) iseed
       do i=1,100
       x=ran1(iseed)
       enddo

       allocate(ind_orig_case(num_case))
       allocate(weight_orig_case(num_case))

       num_select0=num_select

3300   continue
       fact=num_select*1.d0/nfeat2
       iselect=0
       do ii=1,num_case
       x=ran1(iseed)
       if(x.lt.fact*weight_case(ii)) then
       iselect=iselect+1
       ind_orig_case(iselect)=ii
       weight_orig_case(iselect)=weight_case(ii)
       if(iselect.eq.num_select0) goto 4000
       endif
       enddo
       if(iselect.lt.num_select0) then
       num_select=num_select*1.1
       goto 3300
       endif

4000   continue
       num_select=iselect
       write(6,*) "actual num_select=",num_select

       open(12,file="Ind_reference."//char(itype+48))
       rewind(12)
       write(12,*)  num_select
       do i=1,num_select
       write(12,"(2(i6,1x),E15.6)") i,ind_orig_case(i),
     &     weight_orig_case(i)    ! the case index in the original cases
       enddo
       close(12)
100    continue

ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
ccccc Read in the feature again. This time from feat_new_stored (after normalization)
       open(10,file="feat_new_stored."//char(itype+48),
     &    form="unformatted")
       rewind(10)
       read(10) num_case,nfeat2
       write(6,*) "num_case,nfeat", num_case,nfeat2
       do ii=1,num_case
       read(10) jj,Ei_case(ii),feat2_case(:,ii)
       enddo
       close(10)

       num_ref=num_select
       allocate(feat2_ref(nfeat2,num_ref))
       do i=1,num_ref
       do j=1,nfeat2
       feat2_ref(j,i)=feat2_case(j,ind_orig_case(i))
       enddo
       enddo

       open(11,file="feat2_ref."//char(itype+48),form="unformatted")
       rewind(11)
       write(11) nfeat2,num_ref
       write(11) feat2_ref
       close(11)



       stop
       end
       







       
