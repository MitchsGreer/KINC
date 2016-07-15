#include "cmatrix.h"











CMatrix::GPair::~GPair()
{
   if (_iModes)
   {
      delete _iModes;
   }
   if (_iCorrelations)
   {
      delete _iCorrelations;
   }
}



void CMatrix::GPair::read()
{
   _p->File::mem().sync(_data,FSync::read,diagonal(_x,_y));
}



void CMatrix::GPair::write()
{
   _p->File::mem().sync(_data,FSync::write,diagonal(_x,_y));
}



CMatrix::GPair::Modes& CMatrix::GPair::modes()
{
   if (!_iModes)
   {
      _iModes = new Modes(this);
   }
   return *_iModes;
}



CMatrix::GPair::Correlations& CMatrix::GPair::correlations()
{
   if (_iCorrelations)
   {
      _iCorrelations = new Correlations(this);
   }
   return *_iCorrelations;
}



void CMatrix::GPair::operator++()
{
   if ((_y+1)<_x)
   {
      _y++;
   }
   else if (_x<_p->_hdr.gSize())
   {
      _y = 0;
      ++_x;
   }
}



bool CMatrix::GPair::operator!=(const GPair& cmp)
{
   return _p!=cmp._p||_x!=cmp._x||_y!=cmp._y;
}



CMatrix::GPair::GPair(CMatrix* p, int x, int y):
   _p(p),
   _data(p->_hdr.mSize(),p->_hdr.sSize(),p->_hdr.cSize(),p->_hdr.eData()),
   _x(x),
   _y(y)
{}



void CMatrix::GPair::set(int x, int y)
{
   _x = x;
   _y = y;
}



CMatrix::GPair::Modes::~Modes()
{
   if (_iMode)
   {
      delete _iMode;
   }
}



CMatrix::GPair::Modes::Mode CMatrix::GPair::Modes::begin()
{
   return Mode(this,0);
}



CMatrix::GPair::Modes::Mode CMatrix::GPair::Modes::end()
{
   return Mode(this,_p->_p->_hdr.mSize());
}



CMatrix::GPair::Modes::Mode& CMatrix::GPair::Modes::at(int i)
{
   bool cond {i>=0&&i<(_p->_p->_hdr.mSize())};
   AccelCompEng::assert<OutOfRange>(cond,__LINE__);
   return (*this)[i];
}



CMatrix::GPair::Modes::Mode& CMatrix::GPair::Modes::operator[](int i)
{
   if (!_iMode)
   {
      _iMode = new Mode(this,i);
   }
   else
   {
      _iMode->set(i);
   }
   return *_iMode;
}



CMatrix::GPair::Modes::Modes(GPair* p):
   _p(p)
{}



CMatrix::GPair::Modes::Mode::Iterator CMatrix::GPair::Modes::Mode::begin()
{
   return Iterator(this,0);
}



CMatrix::GPair::Modes::Mode::Iterator CMatrix::GPair::Modes::Mode::end()
{
   return Iterator(this,_p->_p->_p->_hdr.sSize());
}



int8_t& CMatrix::GPair::Modes::Mode::at(int i)
{
   bool cond {i>=0&&i<(_p->_p->_p->_hdr.sSize())};
   AccelCompEng::assert<OutOfRange>(cond,__LINE__);
   return (*this)[i];
}



int8_t& CMatrix::GPair::Modes::Mode::operator[](int i)
{
   return _p->_p->_data.mVal(_i,i);
}



void CMatrix::GPair::Modes::Mode::operator++()
{
   if (_i<(_p->_p->_p->_hdr.sSize()))
   {
      ++_i;
   }
}



bool CMatrix::GPair::Modes::Mode::operator!=(const Mode& cmp)
{
   return _p!=cmp._p||_i!=cmp._i;
}



CMatrix::GPair::Modes::Mode::Mode(Modes* p, int i):
   _p(p),
   _i(i)
{}



void CMatrix::GPair::Modes::Mode::set(int i)
{
   _i = i;
}



int8_t& CMatrix::GPair::Modes::Mode::Iterator::operator*()
{
   return _p->_p->_p->_data.mVal(_p->_i,_i);
}



void CMatrix::GPair::Modes::Mode::Iterator::operator++()
{
   if (_i<(_p->_p->_p->_p->_hdr.sSize()))
   {
      ++_i;
   }
}



bool CMatrix::GPair::Modes::Mode::Iterator::operator!=(const Iterator& cmp)
{
   return _p!=cmp._p||_i!=cmp._i;
}



CMatrix::GPair::Modes::Mode::Iterator::Iterator(Mode* p, int i):
   _p(p),
   _i(i)
{}



CMatrix::GPair::Corrs::~Corrs()
{
   if (_iCorr)
   {
      delete _iCorr;
   }
}



CMatrix::GPair::Corrs::Corr CMatrix::GPair::Corrs::begin()
{
   return Corr(this,0);
}



CMatrix::GPair::Corrs::Corr CMatrix::GPair::Corrs::end()
{
   return Corr(this,BLASH)
}
CMatrix::GPair::Corrs::Corr& CMatrix::GPair::Corrs::at(int) {}
CMatrix::GPair::Corrs::Corr& CMatrix::GPair::Corrs::operator[](int) {}
CMatrix::GPair::Corrs::Corrs(GPair*) {}












///////////////////////////////////////////////////////////////////////////////////
CMatrix::CMatrix(const string& type, const string& file):
   DataPlugin(type,file)
{
   if (File::head()==FileMem::nullPtr)
   {
      File::mem().allot(_hdr);
      File::head(_hdr.addr());
      _hdr.gSize() = 0;
      _hdr.sSize() = 0;
      _hdr.cSize() = 0;
      _hdr.mdMax() = 0;
      _hdr.gPtr() = fNullPtr;
      _hdr.sPtr() = fNullPtr;
      _hdr.cPtr() = fNullPtr;
      _hdr.mdData() = fNullPtr;
      _hdr.crData() = fNullPtr;
      File::mem().sync(_hdr,FSync::write);
   }
   else
   {
      _hdr = File::head();
      File::mem().sync(_hdr,FSync::read);
   }
}



void CMatrix::load(GetOpts&, Terminal &tm)
{
   tm << "Not yet implemented.\n";
}



void CMatrix::dump(GetOpts&, Terminal &tm)
{
   tm << "Not yet implemented.\n";
}



void CMatrix::query(GetOpts&, Terminal &tm)
{
   tm << "Not yet implemented.\n";
}



bool CMatrix::empty()
{
   return _hdr.gSize()==0;
}



void CMatrix::initialize(uint32_t gSize, uint32_t sSize, uint32_t cSize,
                         uint8_t mdMax)
{
   bool cond {_hdr.gSize()==0};
   AccelCompEng::assert<AlreadyInitialized>(cond,__LINE__);
   cond = gSize>0&&sSize>0&&cSize>0&&mdMax>0;
   AccelCompEng::assert<InvalidSize>(cond,__LINE__);
   _hdr.gSize() = gSize;
   _hdr.sSize() = sSize;
   _hdr.cSize() = cSize;
   _hdr.mdMax() = mdMax;
   NmHead nh;
   File::mem().allot(nh,gSize);
   _hdr.gPtr() = nh.addr();
   File::mem().allot(nh,sSize);
   _hdr.sPtr() = nh.addr();
   File::mem().allot(nh,cSize);
   _hdr.cPtr() = nh.addr();
   _hdr.mdData() = GeneModes::initialize(*this,gSize,sSize,mdMax);
   _hdr.crData() = GeneCorrs::initialize(*this,gSize,cSize,mdMax);
   File::mem().sync(_hdr,FSync::write);
}



void CMatrix::set_gene_name(uint32_t i, const string& name)
{
   set_name(_hdr.gPtr(),i,_hdr.gSize(),name);
}



void CMatrix::set_sample_name(uint32_t i, const string& name)
{
   set_name(_hdr.sPtr(),i,_hdr.sSize(),name);
}



void CMatrix::set_correlation_name(uint32_t i, const string& name)
{
   set_name(_hdr.cPtr(),i,_hdr.cSize(),name);
}



CMatrix::GeneModes CMatrix::get_modes(uint32_t g1, uint32_t g2)
{
   bool cond {_hdr.mdData()!=fNullPtr};
   AccelCompEng::assert<NotInitialized>(cond,__LINE__);
   return GeneModes(this,_hdr.mdData()+diagonal(g1,g2));
}



void CMatrix::get_modes(GeneModes& gm, uint32_t g1, uint32_t g2)
{
   bool cond {_hdr.mdData()!=fNullPtr};
   AccelCompEng::assert<NotInitialized>(cond,__LINE__);
   gm.addr(_hdr.mdData()+diagonal(g1,g2));
}



CMatrix::GeneCorrs CMatrix::get_corrs(uint32_t g1, uint32_t g2)
{
   bool cond {_hdr.crData()!=fNullPtr};
   AccelCompEng::assert<NotInitialized>(cond,__LINE__);
   return GeneCorrs(this,_hdr.crData()+diagonal(g1,g2));
}



void CMatrix::get_corrs(GeneCorrs& gc, uint32_t g1, uint32_t g2)
{
   bool cond {_hdr.crData()!=fNullPtr};
   AccelCompEng::assert<NotInitialized>(cond,__LINE__);
   gc.addr(_hdr.crData()+diagonal(g1,g2));
}



void CMatrix::set_name(FPtr ptr, uint32_t i, uint32_t size, const string& name)
{
   bool cond {size>0};
   AccelCompEng::assert<NotInitialized>(cond,__LINE__);
   cond = i<size;
   AccelCompEng::assert<OutOfRange>(cond,__LINE__);
   NmHead nm {ptr};
   FString str {&File::mem()};
   str = name;
   nm.nPtr() = str.addr();
   File::mem().sync(nm,FSync::write,i);
}



CMatrix::SizeT CMatrix::diagonal(uint32_t g1, uint32_t g2)
{
   bool cond {g1!=g2};
   AccelCompEng::assert<InvalidGeneCorr>(cond,__LINE__);
   cond = g1<_hdr.gSize()&&g2<_hdr.gSize();
   AccelCompEng::assert<OutOfRange>(cond,__LINE__);
   if (g1<g2)
   {
      uint32_t tmp {g1};
      g1 = g2;
      g2 = tmp;
   }
   return (g1*(g1+1)/2)+g2;
}



CMatrix::SizeT CMatrix::diag_size(uint32_t gNum)
{
   return gNum*(gNum-1)/2;
}



bool CMatrix::GeneModes::mode(uint8_t md, uint32_t i)
{
   bool cond {md<_p->_hdr.mdMax()&&i<md<_p->_hdr.sSize()};
   AccelCompEng::assert<OutOfRange>(cond,__LINE__);
   AccelCompEng::assert<ValueNotRead>(_isRead,__LINE__);
   return _data.mask(md,i);
}



void CMatrix::GeneModes::mode(uint8_t md, uint32_t i, bool val)
{
   bool cond {md<_p->_hdr.mdMax()&&i<md<_p->_hdr.sSize()};
   AccelCompEng::assert<OutOfRange>(cond,__LINE__);
   _data.mask(md,i) = val;
}



void CMatrix::GeneModes::mode(uint8_t md, bool val)
{
   bool cond {md<_p->_hdr.mdMax()};
   AccelCompEng::assert<OutOfRange>(cond,__LINE__);
   for (int i=0;i<_p->_hdr.sSize();++i)
   {
      _data.mask(md,i) = val;
   }
}



void CMatrix::GeneModes::mode(bool val)
{
   for (int md=0;md<_p->_hdr.mdMax();++md)
   {
      for (int i=0;i<_p->_hdr.sSize();++i)
      {
         _data.mask(md,i) = val;
      }
   }
}



void CMatrix::GeneModes::read()
{
   _p->File::mem().sync(_data,FSync::read);
   _isRead = true;
}



void CMatrix::GeneModes::write()
{
   _p->File::mem().sync(_data,FSync::write);
}



CMatrix::GeneModes::GeneModes(CMatrix* p, FPtr ptr):
   _p(p),
   _data(p->_hdr.mdMax(),p->_hdr.sSize(),ptr)
{}



void CMatrix::GeneModes::addr(FPtr ptr)
{
   _data.addr(ptr);
   _isRead = false;
}



CMatrix::FPtr CMatrix::GeneModes::initialize(CMatrix& p, uint32_t gSize,
                                             uint32_t sSize,
                                             uint8_t mdMax)
{
   Modes md(mdMax,sSize);
   p.File::mem().allot(md,p.diag_size(gSize));
   return md.addr();
}



float CMatrix::GeneCorrs::corr(uint8_t md, uint32_t i)
{
   bool cond {md<_p->_hdr.mdMax()&&i<md<_p->_hdr.cSize()};
   AccelCompEng::assert<OutOfRange>(cond,__LINE__);
   AccelCompEng::assert<ValueNotRead>(_isRead,__LINE__);
   return _data.val(md,i);
}



void CMatrix::GeneCorrs::corr(uint8_t md, uint32_t i, float val)
{
   bool cond {md<_p->_hdr.mdMax()&&i<md<_p->_hdr.cSize()};
   AccelCompEng::assert<OutOfRange>(cond,__LINE__);
   _data.val(md,i) = val;
}



void CMatrix::GeneCorrs::read()
{
   _p->File::mem().sync(_data,FSync::read);
   _isRead = true;
}



void CMatrix::GeneCorrs::write()
{
   _p->File::mem().sync(_data,FSync::write);
}



CMatrix::GeneCorrs::GeneCorrs(CMatrix* p, FPtr ptr):
   _p(p),
   _data(p->_hdr.mdMax(),p->_hdr.cSize(),ptr)
{}



void CMatrix::GeneCorrs::addr(FPtr ptr)
{
   _data.addr(ptr);
   _isRead = false;
}



CMatrix::FPtr CMatrix::GeneCorrs::initialize(CMatrix& p, uint32_t gSize,
                                             uint32_t cSize,
                                             uint8_t mdMax)
{
   Corrs cr(mdMax,cSize);
   p.File::mem().allot(cr,p.diag_size(gSize));
   return cr.addr();
}
