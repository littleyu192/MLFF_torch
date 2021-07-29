      subroutine heapsort(v_input,n,ind2)
c-----------------------------------------------------------------------
      implicit none
c
c     .. Scalars
      integer i,ir,j,l,n
      integer tmpref,tmp1,tmp2,tmp3
c
c     .. Arrays
      integer vref(n),ind(n),ind2(n)
      integer v_input(n)
c-----------------------------------------------------------------------
      vref=v_input

      do i=1,n
      ind(i)=i
      enddo


      l=n/2+1
      ir=n
   50 continue
      if (l.gt.1) then
         l=l-1
         tmpref=vref(l)
         tmp1=ind(l)
      else
         tmpref=vref(ir)
         tmp1=ind(ir)
         vref(ir)=vref(1)
         ind(ir)=ind(1)
         ir=ir-1
         if (ir.eq.1) then
            vref(ir)=tmpref
            ind(ir)=tmp1
            goto 70
         endif
      endif
      i=l
      j=l+l
   60 if (j.le.ir) then
         if (j.lt.ir) then
            if (vref(j).lt.vref(j+1)) j=j+1
         endif
         if (tmpref.lt.vref(j)) then
            vref(i)=vref(j)
            ind(i)=ind(j)
            i=j
            j=j+j
         else
            j=ir+1
         endif
         goto 60
      endif
      vref(i)=tmpref
      ind(i)=tmp1
      goto 50
   70 continue

      do i=1,n
      ind2(n+1-i)=ind(i)
      enddo
      
c
      return
      end            
