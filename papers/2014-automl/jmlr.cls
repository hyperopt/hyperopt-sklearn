%%
%% This is file `jmlr.cls',
%% generated with the docstrip utility.
%%
%% The original source files were:
%%
%% jmlr.dtx  (with options: `jmlr.cls,package')
%% 
%%  jmlr.dtx
%%  Copyright 2011 Nicola Talbot
%% 
%%  This work may be distributed and/or modified under the
%%  conditions of the LaTeX Project Public License, either version 1.3
%%  of this license of (at your option) any later version.
%%  The latest version of this license is in
%%    http://www.latex-project.org/lppl.txt
%%  and version 1.3 or later is part of all distributions of LaTeX
%%  version 2005/12/01 or later.
%% 
%%  This work has the LPPL maintenance status `maintained'.
%% 
%%  The Current Maintainer of this work is Nicola Talbot.
%% 
%%  This work consists of the files jmlr.dtx and jmlr.ins and the derived files jmlr.cls, jmlrbook.cls.
%% 
%% \CharacterTable
%%  {Upper-case    \A\B\C\D\E\F\G\H\I\J\K\L\M\N\O\P\Q\R\S\T\U\V\W\X\Y\Z
%%   Lower-case    \a\b\c\d\e\f\g\h\i\j\k\l\m\n\o\p\q\r\s\t\u\v\w\x\y\z
%%   Digits        \0\1\2\3\4\5\6\7\8\9
%%   Exclamation   \!     Double quote  \"     Hash (number) \#
%%   Dollar        \$     Percent       \%     Ampersand     \&
%%   Acute accent  \'     Left paren    \(     Right paren   \)
%%   Asterisk      \*     Plus          \+     Comma         \,
%%   Minus         \-     Point         \.     Solidus       \/
%%   Colon         \:     Semicolon     \;     Less than     \<
%%   Equals        \=     Greater than  \>     Question mark \?
%%   Commercial at \@     Left bracket  \[     Backslash     \\
%%   Right bracket \]     Circumflex    \^     Underscore    \_
%%   Grave accent  \`     Left brace    \{     Vertical bar  \|
%%   Right brace   \}     Tilde         \~}
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{jmlr}[2011/03/24 v1.11 (NLCT) Journal of Machine Learning Research]
\RequirePackage{xkeyval}
\providecommand*{\jmlr@prehyperref}{}
\define@key{jmlr.cls}{prehyperref}{%
  \renewcommand*{\jmlr@prehyperref}{#1}}
\newif\if@openright
\newif\if@mainmatter \@mainmattertrue
\@ifundefined{ifgrayscale}{
  \newif\ifgrayscale
  \grayscalefalse
}{}
\DeclareOptionX{color}{\grayscalefalse
  \PassOptionsToPackage{color}{xcolor}}
\DeclareOptionX{gray}{\grayscaletrue
  \PassOptionsToPackage{gray}{xcolor}}
\newif\iftablecaptiontop
\tablecaptiontoptrue
\DeclareOptionX{tablecaptiontop}{\tablecaptiontoptrue}
\DeclareOptionX{tablecaptionbottom}{\tablecaptiontopfalse}

\define@choicekey{jmlr.cls}{tablecaption}[\val\nr]{top,bottom}{%
  \ifcase\nr\relax
    \tablecaptiontoptrue
  \or
    \tablecaptiontopfalse
  \fi
}
\newif\ifjmlrhtml
\jmlrhtmlfalse
\DeclareOptionX{html}{\jmlrhtmltrue}
\DeclareOptionX{nohtml}{\jmlrhtmlfalse}
\def\pt@size{11pt}
\DeclareOptionX{10pt}{\renewcommand{\pt@size}{10pt}}
\DeclareOptionX{11pt}{\renewcommand{\pt@size}{11pt}}
\DeclareOptionX{12pt}{\renewcommand{\pt@size}{12pt}}
\newcommand*{\@jmlrproceedings}{Journal of Machine Learning Research}
\newcommand*{\@jmlrabbrvproceedings}{JMLR}
\newcommand*{\jmlrproceedings}[2]{%
  \renewcommand*{\@jmlrabbrvproceedings}{#1}%
  \renewcommand*{\@jmlrproceedings}{#2}%
}
\newcommand*{\jmlrnowcp}{%
  \jmlrproceedings{JMLR}{Journal of Machine Learning Research}%
}
\newcommand*{\jmlrwcp}{%
  \jmlrproceedings{JMLR W\&CP}{}%
}
\DeclareOptionX{nowcp}{\jmlrnowcp}
\DeclareOptionX{wcp}{\jmlrwcp}
\newif\ifviiXx
\viiXxfalse
\DeclareOptionX{7x10}{\viiXxtrue}
\DeclareOptionX{letterpaper}{\PassOptionsToPackage{letterpaper}{typearea}}
\DeclareOptionX*{\PassOptionsToClass{\CurrentOption}{article}}
\ExecuteOptions{twoside,letterpaper}
\ProcessOptionsX
\LoadClass[\pt@size]{article}
\ifviiXx
  \setlength{\paperwidth}{7in}
  \setlength{\paperheight}{10in}
  \setlength{\textwidth}{5.25in}
  \setlength{\textheight}{8.2in}
  \setlength{\topmargin}{0.4in}
  \setlength{\headheight}{0.2in}
  \setlength{\headsep}{0.2in}
  \setlength{\hoffset}{-1in}
  \setlength{\voffset}{-1in}
  \setlength{\evensidemargin}{0.75in}
  \setlength{\oddsidemargin}{1.0in}
\else
  \setlength{\oddsidemargin}{0.25in}
  \setlength{\evensidemargin}{0.25in}
  \setlength{\marginparwidth}{0.07 true in}
  \setlength{\topmargin}{-0.5in}
  \addtolength{\headsep}{0.25in}
  \setlength{\textheight}{8.5 true in}
  \setlength{\textwidth}{6.0 true in}
\fi
\AtEndDocument{\@jmlrenddoc}
\RequirePackage{amsmath}
\RequirePackage{amssymb}
\RequirePackage{natbib}
\RequirePackage{graphicx}
\RequirePackage{url}
\RequirePackage[x11names]{xcolor}
\RequirePackage[algo2e,ruled]{algorithm2e}
\jmlr@prehyperref
\usepackage[colorlinks,
            linkcolor=blue,
            citecolor=blue,
            urlcolor=magenta,
            linktocpage,
            plainpages=false]{hyperref}
\usepackage{nameref}
\ifgrayscale
  \hypersetup{draft}
\fi
\renewcommand{\topfraction}{0.95} % let figure take up nearly whole page
\renewcommand{\textfraction}{0.05} % let figure take up nearly whole page
\widowpenalty=10000\relax
\clubpenalty=10000\relax
\@twosidetrue
\@mparswitchtrue
\def\ds@draft{\overfullrule 5pt}
\bibliographystyle{plainnat}
\bibpunct{(}{)}{;}{a}{,}{,}
\renewcommand{\section}{\@startsection{section}{1}{\z@}%
   {-0.24in \@plus -1ex \@minus -.2ex}%
   {0.10in \@plus.2ex}%
   {\normalfont\rmfamily\bfseries\large\raggedright}%
}
\renewcommand\subsection{\@startsection{subsection}{2}{\z@}%
   {-0.20in \@plus -1ex \@minus -.2ex}%
   {0.08in \@plus .2ex}%
   {\normalfont\rmfamily\bfseries\normalsize\raggedright}%
}
\renewcommand\subsubsection{\@startsection{subsubsection}{3}{\z@}%
   {-0.18in \@plus -1ex \@minus -.2ex}%
   {0.08in \@plus .2ex}%
   {\normalfont\normalsize\rmfamily\mdseries\scshape\raggedright}%
}
\renewcommand\paragraph{\@startsection{paragraph}{4}{\z@}%
   {1.5ex plus 0.5ex minus .2ex}%
   {-1em}%
   {\normalfont\normalsize\rmfamily\bfseries}%
}
\renewcommand\subparagraph{\@startsection{subparagraph}{5}{\z@}%
   {1.5ex plus 0.5ex minus .2ex}%
   {-1em}%
   {\normalfont\normalsize\rmfamily\bfseries\itshape}}
\renewcommand*\@seccntformat[1]{%
  \csname pre#1num\endcsname
  \csname the#1\endcsname.\enskip
}
\renewcommand*{\@makefntext}[1]{%
  \@setpar
  {%
    \@@par
    \@tempdima\hsize
    \advance \@tempdima -15pt\relax
    \parshape \@ne 15pt \@tempdima
  }%
  \par
  \parindent 2em\noindent
  \hbox to \z@ {\hss {\@thefnmark }\footnoteseptext\hfil }#1%
}
\newcommand*{\footnoteseptext}{. }
\renewcommand*{\thanks}[1]{%
  \footnotemark
  \protected@xdef\@thanks{\@thanks
    \protect\footnotetext{#1}}%
}
\ifjmlrhtml
  \renewenvironment{abstract}{\HCode{<h3>}Abstract\HCode{</h3>}}{}%
\else
  \renewenvironment{abstract}
  {{\centering\large\bfseries Abstract\par}\vspace{0.7ex}%
    \bgroup
       \leftskip 20pt\rightskip 20pt\small\noindent\ignorespaces}%
  {\par\egroup\vskip 0.25ex}
\fi
\newenvironment{keywords}
{\bgroup\leftskip 20pt\rightskip 20pt \small\noindent{\bfseries
Keywords:} \ignorespaces}%
{\par\egroup\vskip 0.25ex}
\newlength\aftertitskip     \newlength\beforetitskip
\newlength\interauthorskip  \newlength\aftermaketitskip
%% Changeable parameters.
\setlength\aftertitskip{0.1in plus 0.2in minus 0.2in}
\setlength\beforetitskip{0.05in plus 0.08in minus 0.08in}
\setlength\interauthorskip{0.08in plus 0.1in minus 0.1in}
\setlength\aftermaketitskip{0.3in plus 0.1in minus 0.1in}
\renewcommand*{\title}[2][\@title]{%
  \def\@shorttitle{#1}%
  \def\@title{#2}%
  \jmlrtitlehook
}
\newcommand*{\jmlrtitlehook}{}
\renewcommand*{\author}[2][]{%
  \def\@author{#2}%
  \def\@sauthor{#1}%
  \ifx\@sauthor\@empty
  \else
    \let\@shortauthor\@sauthor
  \fi
  \jmlrauthorhook
}
\newcommand*{\jmlrauthorhook}{}
\newcommand*{\@shortauthor}{}
\newcommand*{\@firstauthor}{}
\newcommand*{\@firstsurname}{}
\newlength\jmlrlength
\def\jmlrmaketitle{%
 \jmlrpremaketitlehook
 \def\@jmlr@authors@sep{, }%
 \par
 \begingroup
   \def\footnoteseptext{ }%
   \def\thempfn{\textsuperscript{\thefootnote}}%
   \def\thefootnote{\fnsymbol{footnote}}%
   \if@twocolumn
     \twocolumn[\@jmlrmaketitle]%
   \else
     \@jmlrmaketitle
   \fi
   \@thanks
 \endgroup
\label{jmlrstart}%
\ifx\@sauthor\@empty
  \settowidth{\jmlrlength}{\@evenhead}%
  \ifdim\jmlrlength>\textwidth
    \def\@shortauthor{\@firstsurname\space et al.}%
  \fi
\fi
\settowidth{\jmlrlength}{\@titlefoot}%
\ifdim\jmlrlength>\textwidth
  \def\@jmlrauthors{\@firstauthor\space \emph{et al}}%
\fi
\jmlrmaketitlehook
\thispagestyle{jmlrtps}%
\setcounter{footnote}{0}%
\let\maketitle\relax \let\@maketitle\relax
\gdef\@thanks{}\gdef\@author{}\let\thanks\@gobble
\def\@jmlr@authors@sep{ \& }%
}
\newcommand*{\jmlrmaketitlehook}{}
\newcommand*{\jmlrpremaketitlehook}{}
\newcommand{\jmlrhtmlmaketitle}{%
  \ifx\@jmlrauthors\@empty
    \sbox\jmlrbox{\let\addr\relax\@author}%
  \fi
  \noindent\HCode{<h2>}\@title\HCode{</h2>}
  \noindent\@jmlrauthors
}
%%\begin{macro}{\jmlrbox}
\newsavebox\jmlrbox
\ifjmlrhtml
  \let\maketitle\jmlrhtmlmaketitle
\else
  \let\maketitle\jmlrmaketitle
\fi
\def\@startauthor{\noindent \normalsize\bfseries}
\def\@endauthor{}
\def\jmlrpretitle{\vskip\beforetitskip\begin{center}\Large\bfseries}
\def\jmlrposttitle{\par\end{center}\vskip\aftertitskip}
\newcommand*{\nametag}[1]{}
\def\jmlrpreauthor{%
\bgroup
  \def\nametag##1{##1}%
  \def\and{\unskip\enspace{\normalfont and}\enspace}%
  \def\addr{\mdseries\small\itshape}%
  \def\name{\ClassError{jmlr}{Use \string\Name{Author's Name} not \string\name}{}}%
  \def\email{\ClassError{jmlr}{Use \string\Email{address} not \string\email}{}}%
  \def\AND{\@endauthor\normalfont\hss \vskip \interauthorskip
    \@startauthor}%
  \@startauthor
}
\def\@email{\hfill\small\mdseries\scshape}%
\def\@name{\normalsize\upshape\bfseries}%
\def\@parsename#1 #2\end@parsename{%
  \def\@tmp{#2}%
  \ifx\@tmp\@nnil
    \def\@surname{#1}%
    \let\@nextparsename\@parsenamenoop
  \else
    \@getinitial#1-\relax\relax\end@getinitial
    \ifx\@forenames\@empty
      \def\@forenames{#1}%
      \protected@edef\@initials{\@initial}%
    \else
      \expandafter\toks@\expandafter{\@forenames}%
      \edef\@forenames{\space\the\toks@}%
      \expandafter\toks@\expandafter{\@initials}%
      \protected@edef\@initials{\the\toks@\@initial}%
    \fi
    \let\@nextparsename\@parsename
  \fi
  \@nextparsename#2\end@parsename
}
\def\@parsenamenoop#1\end@parsename{}
\def\@getinitial#1#2-#3#4\end@getinitial{%
  \def\@jmlr@tmp{#3}%
  \if\@jmlr@tmp\relax
    \def\@initial{#1.}%
  \else
    \def\@initial{#1.-#3.}%
  \fi
}
\newcommand*{\Name}[1]{%
  \def\@forenames{}%
  \def\@surname{}%
  \def\nametag##1{}%
  \@parsename#1 \@nil\end@parsename
  \ifx\@shortauthor\@empty
    \ifx\@sauthor\@empty
      \global\let\@shortauthor\@surname
      \global\let\@firstsurname\@surname
    \fi
    \protected@xdef\@jmlrauthors{\@initials\space\@surname}%
    \global\let\@firstauthor\@jmlrauthors
  \else
    \ifx\@sauthor\@empty
      \expandafter\toks@\expandafter{\@shortauthor}%
      \protected@xdef\@shortauthor{\the\toks@\space\@surname}%
    \fi
    \protected@xdef\@jmlrauthors{\@jmlrauthors
      \noexpand\@jmlr@authors@sep
      \@initials\space\@surname}%
  \fi
  \def\nametag##1{##1}%
  \@name #1%
}
\newcommand*{\jmlrabbrnamelist}[1]{%
  \def\nametag##1{}%
  \def\@jmlr@authors@sep{, }%
  \def\@jmlr@namelist{}%
  \@for\@thisname:=#1\do{%
    \expandafter\@jmlrabbrname\expandafter{\@thisname}%
    \ifx\@jmlr@namelist\@empty
       \protected@edef\@jmlr@namelist{%
          \@initials\space\@surname
       }%
    \else
       \protected@edef\@jmlr@namelist{%
          \@jmlr@namelist
          \noexpand\@jmlr@authors@sep
          \@initials\space\@surname
       }%
    \fi
  }%
  \def\@jmlr@authors@sep{ \& }%
  \@jmlr@namelist
}
\newcommand*{\@jmlrabbrname}[1]{%
  \def\@initials{}%
  \def\@surname{}%
  \def\@forenames{}%
  \@parsename#1 \@nil\end@parsename
}
\newcommand*{\Email}[1]{{\@email #1}}
\def\jmlrpostauthor{\@endauthor\egroup
  \par
  \vskip \aftermaketitskip
  \noindent
  \vskip \aftermaketitskip
}
\def\@jmlrmaketitle{\vbox{\hsize\textwidth
 \linewidth\hsize
 \jmlrpretitle \@title \jmlrposttitle
 \jmlrpreauthor \@author \jmlrpostauthor
}}
\newcommand*\kernelmachines{(for
  {\textsc{http://www.kernel-machines.org}})}

\def\firstpageno#1{\setcounter{page}{#1}}
\@ifundefined{startpage}{}{\firstpageno{\startpage}}
\newcommand*{\@jmlrenddoc}{%
  \phantomsection
  \protected@edef\@currentlabelname{end of \@shorttitle}%
  \label{jmlrend}\null
  \global\let\@reprint\@empty
}
\newcommand*{\@titlefoot}{\scriptsize\copyright\space\@jmlryear
    \space\@jmlrauthors.\hfill
    \@reprint
}
\let\@reprint\@empty
\newcommand{\reprint}[1]{%
  \gdef\@reprint{Reprinted with permission for JMLR#1}}
\newcommand\ps@jmlrtps{%
  \let\@mkboth\@gobbletwo
  \def\@oddhead{\scriptsize \@jmlrproceedings
    %\ifx\@jmlrvolume\@empty\else\space\@jmlrvolume\fi
    %\ifx\@jmlryear\@empty\else\space(\@jmlryear)\fi
    %\ifx\@jmlrpages\@empty\else\space\@jmlrpages\fi
    \hfill
    %
    ICML 2014 AutoML Workshop
    %\ifx\@jmlrworkshop\@empty
    %  \ifx\@jmlrsubmitted\@empty
    %  \else
    %    Submitted \@jmlrsubmitted
    %    \ifx\@jmlrpublished\@empty\else;\fi
    %  \fi
    %  \ifx\@jmlrpublished\@empty
    %  \else
    %    \space Published \@jmlrpublished
    %  \fi
    %\else
    %  \space\@jmlrworkshop
    %\fi
  }%
  \let\@evenhead\@oddhead
  \def\@oddfoot{\@titlefoot}%
  \let\@evenfoot\@oddfoot
}
\def\ps@jmlrps{%
  \let\@mkboth\@gobbletwo
  \def\@oddhead{\hfill {\small\scshape \@shorttitle} \hfill}%
  \def\@oddfoot{\hfill \small\rmfamily \thepage \hfill}%
  \def\@evenhead{\hfill {\small\scshape \@shortauthor} \hfill}%
  \def\@evenfoot{\hfill \small\rmfamily \thepage \hfill}%
}%
\pagestyle{jmlrps}
\let\@jmlrvolume\@empty
\newcommand*{\jmlrvolume}[1]{\renewcommand*{\@jmlrvolume}{#1}}
\let\@jmlrissue\@empty
\newcommand*{\jmlrissue}[1]{\renewcommand*{\@jmlrissue}{#1}}
\let\@jmlryear\@empty
\newcommand*{\jmlryear}[1]{\renewcommand*{\@jmlryear}{#1}}
\newcommand*\@jmlrpages{\pageref{jmlrstart}--\pageref{jmlrend}}
\newcommand*{\jmlrpages}[1]{\renewcommand*{\@jmlrpages}{#1}}
\let\@jmlrsubmitted\@empty
\newcommand*{\jmlrsubmitted}[1]{\renewcommand*{\@jmlrsubmitted}{#1}}
\let\@jmlrpublished\@empty
\newcommand*{\jmlrpublished}[1]{\renewcommand*{\@jmlrpublished}{#1}}
\let\@jmlrworkshop\@empty
\newcommand*{\jmlrworkshop}[1]{\renewcommand*{\@jmlrworkshop}{#1}}
\newcommand*{\@jmlrauthors}{}
\newcommand*{\jmlrauthors}[1]{\renewcommand*{\@jmlrauthors}{#1}}

\def\figurecaption#1#2{\noindent\hangindent 40pt
                       \hbox to 36pt {\small\slshape #1 \hfil}
                       \ignorespaces {\small #2}}
\def\figurecenter#1#2{\centerline{{\slshape #1} #2}}
\def\figurecenter#1#2{\centerline{{\small\slshape #1} {\small #2}}}
\long\def\@makecaption#1#2{%
   \vskip 10pt
   \setbox\@tempboxa\hbox{#1: #2}%
   \ifdim \wd\@tempboxa >\hsize               % IF longer than one line:
       \begin{list}{#1:}{%
       \settowidth{\labelwidth}{#1:}
       \setlength{\leftmargin}{\labelwidth}
       \addtolength{\leftmargin}{\labelsep}
        }\item #2 \end{list}\par   % Output in quote mode
     \else                                    %   ELSE  center.
       \hbox to\hsize{\hfil\box\@tempboxa\hfil}
   \fi}
\def\abovestrut#1{\rule[0in]{0in}{#1}\ignorespaces}
\def\belowstrut#1{\rule[-#1]{0in}{#1}\ignorespaces}
\long\def\acks#1{\section*{Acknowledgments}#1}
\long\def\researchnote#1{\noindent {\LARGE\itshape Research Note} #1}
\newcommand*{\set}[1]{\ensuremath{\mathcal{#1}}}
\newcommand*{\@jmlr@reflistsep}{, }
\newcommand*{\@jmlr@reflistlastsep}{ and }
\newcommand*{\sectionrefname}{Section}
\newcommand*{\sectionsrefname}{Sections}
\newcommand*{\equationrefname}{Equation}
\newcommand*{\equationsrefname}{Equations}
\newcommand*{\tablerefname}{Table}
\newcommand*{\tablesrefname}{Tables}
\newcommand*{\figurerefname}{Figure}
\newcommand*{\figuresrefname}{Figures}
\newcommand*{\algorithmrefname}{Algorithm}
\newcommand*{\algorithmsrefname}{Algorithms}
\newcommand*{\theoremrefname}{Theorem}
\newcommand*{\theoremsrefname}{Theorems}
\newcommand*{\lemmarefname}{Lemma}
\newcommand*{\lemmasrefname}{Lemmas}
\newcommand*{\remarkrefname}{Remark}
\newcommand*{\remarksrefname}{Remarks}
\newcommand*{\corollaryrefname}{Corollary}
\newcommand*{\corollarysrefname}{Corollaries}
\newcommand*{\definitionrefname}{Definition}
\newcommand*{\definitionsrefname}{Definitions}
\newcommand*{\conjecturerefname}{Conjecture}
\newcommand*{\conjecturesrefname}{Conjectures}
\newcommand*{\axiomrefname}{Axiom}
\newcommand*{\axiomsrefname}{Axioms}
\newcommand*{\examplerefname}{Example}
\newcommand*{\examplesrefname}{Examples}
\newcommand*{\appendixrefname}{Appendix}
\newcommand*{\appendixsrefname}{Appendices}
\newcommand*{\partrefname}{Part}
\newcommand*{\partsrefname}{Parts}
\DeclareRobustCommand*{\objectref}[5]{%
  \let\@objectname\@empty
  \def\@objectref{}%
  \let\@prevsep\@empty
  \@for\@thislabel:=#1\do{%
    \toks@{\@prevsep}%
    \protected@edef\@objectref{\@objectref\the\toks@
      #4\ref{\@thislabel}#5}%
    \ifx\@objectname\@empty
      \let\@objectname#2% singular tag
    \else
      \let\@objectname#3% plural tag
      \let\@prevsep\@jmlr@reflistsep
    \fi
  }%
  \ifx\@objectname#3% plural tag
    \let\@prevsep\@jmlr@reflistlastsep
  \fi
  \@objectname~\@objectref
}
\newcommand*{\sectionref}[1]{%
  \objectref{#1}{\sectionrefname}{\sectionsrefname}{}{}}
\newcommand*{\equationref}[1]{%
  \objectref{#1}{\equationrefname}{\equationsrefname}()}
\newcommand*{\tableref}[1]{%
  \objectref{#1}{\tablerefname}{\tablesrefname}{}{}}
\newcommand*{\figureref}[1]{%
  \objectref{#1}{\figurerefname}{\figuresrefname}{}{}}
\newcommand*{\algorithmref}[1]{%
  \objectref{#1}{\algorithmrefname}{\algorithmsrefname}{}{}}
\newcommand*{\theoremref}[1]{%
  \objectref{#1}{\theoremrefname}{\theoremsrefname}{}{}}
\newcommand*{\lemmaref}[1]{%
  \objectref{#1}{\lemmarefname}{\lemmasrefname}{}{}}
\newcommand*{\remarkref}[1]{%
  \objectref{#1}{\remarkrefname}{\remarksrefname}{}{}}
\newcommand*{\corollaryref}[1]{%
  \objectref{#1}{\corollaryrefname}{\corollarysrefname}{}{}}
\newcommand*{\definitionref}[1]{%
  \objectref{#1}{\definitionrefname}{\definitionsrefname}{}{}}
\newcommand*{\conjectureref}[1]{%
  \objectref{#1}{\conjecturerefname}{\conjecturesrefname}{}{}}
\newcommand*{\axiomref}[1]{%
  \objectref{#1}{\axiomrefname}{\axiomsrefname}{}{}}
\newcommand*{\exampleref}[1]{%
  \objectref{#1}{\examplerefname}{\examplesrefname}{}{}}
\newcommand*{\appendixref}[1]{%
  \objectref{#1}{\appendixrefname}{\appendixsrefname}{}{}}
\newcommand*{\partref}[1]{%
  \objectref{#1}{\partrefname}{\partsrefname}{}{}}
\newcommand{\floatconts}[3]{%
  \@ifundefined{\@captype conts}{\tableconts{#1}{#2}{#3}}%
  {\csname\@captype conts\endcsname{#1}{#2}{#3}}%
}
\newcommand{\tableconts}[3]{%
  \iftablecaptiontop
    #2\label{#1}\vskip\baselineskip
    {\centering #3\par}%
  \else
    {\centering #3\par}%
    \vskip\baselineskip
    #2\label{#1}%
  \fi
}
\newcommand{\figureconts}[3]{%
  {\centering #3\par}%
  \vskip\baselineskip
  #2\label{#1}%
}
\newcommand{\algocfconts}[3]{%
  \@algocf@pre@ruled
  #2\label{#1}\kern2pt\hrule height.8pt depth0pt\kern2pt%
  #3\@algocf@pre@ruled
}
\newcommand*{\includeteximage}[2][]{%
  \def\Gin@req@sizes{%
    \Gin@req@height\Gin@nat@height
    \Gin@req@width\Gin@nat@width}%
  \begingroup
    \@tempswafalse
    \let\input@path\Ginput@path
    \toks@{\InputIfFileExists{#2}{}{\@warning{File `#1' not found}}}%
    \setkeys{Gin}{#1}%
    \Gin@esetsize
    \the\toks@
  \endgroup
}
\providecommand{\ifprint}[2]{\ifgrayscale#1\else#2\fi}
%%    \begin{macrocode}
\ifjmlrhtml
\else
  \let\@org@Ginclude@graphics\Ginclude@graphics
  \def\Ginclude@graphics#1{%
    \begingroup
    \let\input@path\Ginput@path
    \ifprint{\filename@parse{#1-gray}}{\filename@parse{#1}}%
    \ifx\filename@ext\relax
      \@for\Gin@temp:=\Gin@extensions\do{%
        \ifx\Gin@ext\relax
          \Gin@getbase\Gin@temp
        \fi}%
    \else
      \ifprint{\filename@parse{#1}}{}%
      \Gin@getbase{\Gin@sepdefault\filename@ext}%
      \ifx\Gin@ext\relax
         \@warning{File `#1' not found}%
         \def\Gin@base{\filename@area\filename@base}%
         \edef\Gin@ext{\Gin@sepdefault\filename@ext}%
      \fi
    \fi
      \ifx\Gin@ext\relax
        \ifprint{\@org@Ginclude@graphics{#1}}%
        {%
           \@latex@error{File `#1' not found}%
           {I could not locate the file with any of these extensions:^^J%
            \Gin@extensions^^J\@ehc}%
        }%
      \else
         \@ifundefined{Gin@rule@\Gin@ext}%
           {\ifx\Gin@rule@*\@undefined
              \@latex@error{Unknown graphics extension: \Gin@ext}\@ehc
            \else
              \expandafter\Gin@setfile\Gin@rule@*{\Gin@base\Gin@ext}%
             \fi}%
           {\expandafter\expandafter\expandafter\Gin@setfile
               \csname Gin@rule@\Gin@ext\endcsname{\Gin@base\Gin@ext}}%
      \fi
    \endgroup}
\fi
\newenvironment{algorithm}[1][htbp]%
{%
  \begin{algocf}[#1]%
  \renewcommand\@makecaption[2]{%
    \hskip\AlCapHSkip
    \parbox[t]{\hsize}{\algocf@captiontext{##1}{##2}}%
  }%
}%
{%
  \end{algocf}%
}
\setlength\algomargin{0pt}
\newcommand{\artappendix}{\par
  \setcounter{section}{0}
  \setcounter{subsection}{0}
  \def\thesection{\Alph{section}}
  \def\theHsection{\Alph{section}}
  \def\presectionnum{Appendix~}%
}
\let\appendix\artappendix
\newcommand{\BlackBox}{\rule{1.5ex}{1.5ex}}
\newenvironment{proof}{\par\noindent{\bfseries\upshape
  Proof\ }}{\hfill\BlackBox\\[2mm]}
\newtheorem{example}{Example}
\newtheorem{theorem}{Theorem}
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{remark}[theorem]{Remark}
\newtheorem{corollary}[theorem]{Corollary}
\newtheorem{definition}[theorem]{Definition}
\newtheorem{conjecture}[theorem]{Conjecture}
\newtheorem{axiom}[theorem]{Axiom}
\renewcommand*{\vec}[1]{\boldsymbol{#1}}
\newenvironment{enumerate*}%
{%
  \ifnum\@enumdepth=0\relax
    \setcounter{enumi}{0}%
  \fi
  \ifnum\@enumdepth>\thr@@
    \@toodeep
  \else
    \advance\@enumdepth\@ne
    \def\@enumctr{enumi}%
    \list
      {\labelenumi}%
      {\@nmbrlisttrue\def\@listctr{enumi}%
       \def\makelabel##1{\hss\llap{##1}}}%
  \fi
}%
{\endlist}
\newenvironment{altdescription}[1]%
  {\list{}%
    {%
      \settowidth{\labelwidth}{\altdescriptionlabel{#1}}%
      \setlength{\labelsep}{15pt}%
      \setlength{\leftmargin}{2\labelsep}%
      \addtolength{\leftmargin}{\labelwidth}%
      \setlength{\rightmargin}{\labelsep}%
      \let\makelabel\altdescriptionlabel
    }%
  }%
  {\endlist}

\newcommand*{\altdescriptionlabel}[1]{\normalfont\bfseries #1\hfill}
\newcommand*{\mailto}[1]{\texttt{#1}}
\newcounter{subfigure}
\@addtoreset{subfigure}{figure}
\renewcommand*{\thesubfigure}{\alph{subfigure}}
\renewcommand*{\p@subfigure}{\expandafter\@p@subfigure}
\newcommand*{\@p@subfigure}[1]{%
  \protect\@subfigurelabel{\thefigure}{\thesubfigure}%
}
\newcommand*\@subfigurelabel[2]{#1\subfigurelabel{#2}}
\newcommand*\@subfigref[1]{%
  {%
    \def\@subfigurelabel##1##2{\subfigurelabel{##2}}%
    \ref{#1}%
  }%
}
\newcommand*{\subfigref}[1]{%
  \let\@objectname\@empty
  \def\@objectref{}%
  \let\@prevsep\@empty
  \@for\@thislabel:=#1\do{%
    \toks@{\@prevsep}%
    \protected@edef\@objectref{\@objectref\the\toks@
      \protect\@subfigref{\@thislabel}}%
      \ifx\@objectname\@empty
\let\@objectname\@nil
      \else
\let\@objectname\relax
        \let\@prevsep\@jmlr@reflistsep
      \fi
  }%
  \ifx\@objectname\relax
    \let\@prevsep\@jmlr@reflistlastsep
  \fi
  \@objectref
}
\newcommand*{\subfigurelabel}[1]{(\emph{#1})}
\newsavebox\@subfloatcapbox
\newsavebox\@subfloatcontsbox
\newcommand*{\subfigure}[1][]{%
  \bgroup
  \def\@subfigcap{#1}%
  \@subfigure
}
\newcommand*{\@subfigure}[2][b]{%
  \advance\c@figure by 1\relax
  \refstepcounter{subfigure}%
  \sbox\@subfloatcapbox{\subfigurelabel{\thesubfigure}%
  \ifx\@subfigcap\@empty
  \else
    \space\@subfigcap
  \fi}%
  \sbox\@subfloatcontsbox{#2}%
  \settowidth{\@tempdima}{\usebox\@subfloatcontsbox}%
  \settowidth{\@tempdimb}{\usebox\@subfloatcapbox}%
  \ifdim\@tempdimb>\@tempdima
    \settowidth\@tempdimb{\subfigurelabel{\thesubfigure}\space}%
    \addtolength{\@tempdima}{-\@tempdimb}%
    \sbox\@subfloatcapbox{\subfigurelabel{\thesubfigure}\space
      \parbox[t]{\@tempdima}{\@subfigcap}}%
  \fi
  \begin{tabular}[#1]{@{}c@{}}%
  \usebox\@subfloatcontsbox\\\usebox\@subfloatcapbox
  \end{tabular}%
  \egroup
}
\newcounter{subtable}
\@addtoreset{subtable}{table}
\renewcommand*{\thesubtable}{\alph{subtable}}
\renewcommand*{\p@subtable}{\expandafter\@p@subtable}
\newcommand*{\@p@subtable}[1]{%
  \protect\@subtablelabel{\thetable}{\thesubtable}%
}
\newcommand*\@subtablelabel[2]{#1\subtablelabel{#2}}
\newcommand*\@subtabref[1]{%
  {%
    \def\@subtablelabel##1##2{\subtablelabel{##2}}%
    \ref{#1}%
  }%
}
\newcommand*{\subtabref}[1]{%
  \let\@objectname\@empty
  \def\@objectref{}%
  \let\@prevsep\@empty
  \@for\@thislabel:=#1\do{%
    \toks@{\@prevsep}%
    \protected@edef\@objectref{\@objectref\the\toks@
      \protect\@subtabref{\@thislabel}}%
      \ifx\@objectname\@empty
\let\@objectname\@nil
      \else
\let\@objectname\relax
        \let\@prevsep\@jmlr@reflistsep
      \fi
  }%
  \ifx\@objectname\relax
    \let\@prevsep\@jmlr@reflistlastsep
  \fi
  \@objectref
}
\newcommand*{\subtablelabel}[1]{(\emph{#1})}
\newcommand*{\subtable}[1][]{%
  \def\@subtabcap{#1}%
  \@subtable
}
\newcommand*{\@subtable}[2][t]{%
  \refstepcounter{subtable}%
  \sbox\@subfloatcapbox{\subtablelabel{\thesubtable}%
  \ifx\@subtabcap\@empty
  \else
    \space\@subtabcap
  \fi}%
  \sbox\@subfloatcontsbox{#2}%
  \settowidth{\@tempdima}{\usebox\@subfloatcontsbox}%
  \settowidth{\@tempdimb}{\usebox\@subfloatcapbox}%
  \ifdim\@tempdimb>\@tempdima
    \settowidth\@tempdimb{\subtablelabel{\thesubtable}\space}%
    \addtolength{\@tempdima}{-\@tempdimb}%
    \sbox\@subfloatcapbox{\subtablelabel{\thesubtable}\space
      \parbox[t]{\@tempdima}{\@subtabcap}}%
  \fi
  \begin{tabular}[#1]{@{}c@{}}%
  \usebox\@subfloatcapbox\\\usebox\@subfloatcontsbox
  \end{tabular}
}
\newcounter{chapter}
\renewcommand\thechapter{\@arabic\c@chapter}
\newcommand\@chapapp{\chaptername}
\@addtoreset{section}{chapter}
\newcommand*\chaptermark[1]{}
\newcommand\bookchapter{%
  \if@openright\cleardoublepage\else\clearpage\fi
                    \thispagestyle{plain}%
                    \global\@topnum\z@
                    \@afterindentfalse
                    \secdef\@chapter\@schapter}
\newcommand\artchapter{%
  \ClassError{jmlr}{Chapters not permitted in articles}{}}
\let\chapter\artchapter
\def\@chaptoclabel{chapter}
\def\@chapter[#1]#2{\ifnum \c@secnumdepth >\m@ne
                       \refstepcounter{chapter}%
                       \if@mainmatter
                         \typeout{\@chapapp\space\thechapter.}%
                         \addcontentsline{toc}{\@chaptoclabel}%
                                   {\protect\numberline{\thechapter}#1}%
                       \else
                         \addcontentsline{toc}{\@chaptoclabel}{#1}%
                       \fi
                    \else
                      \addcontentsline{toc}{\@chaptoclabel}{#1}%
                    \fi
                    \chaptermark{#1}%
                    \addtocontents{lof}{\protect\addvspace{10\p@}}%
                    \addtocontents{lot}{\protect\addvspace{10\p@}}%
                    \if@twocolumn
                      \@topnewpage[\@makechapterhead{#2}]%
                    \else
                      \@makechapterhead{#2}%
                      \@afterheading
                    \fi}
\newcommand{\chaptertitleformat}[1]{%
  \Huge\bfseries#1%
}
\newcommand{\chapternumberformat}[1]{%
  \huge\bfseries \@chapapp\space#1\par\nobreak
  \vskip 20\p@
}
\newcommand*{\chapterformat}{\raggedright}
\newlength\postchapterskip
\setlength\postchapterskip{40pt}
\newlength\prechapterskip
\setlength\prechapterskip{50pt}
\def\@makechapterhead#1{%
  \null\vskip\prechapterskip
  {\parindent \z@ \normalfont\chapterformat
    \ifnum \c@secnumdepth >\m@ne
      \if@mainmatter
        \chapternumberformat{\thechapter}%
      \fi
    \fi
    \interlinepenalty\@M
    \chaptertitleformat{#1}\par\nobreak
    \vskip \postchapterskip
  }}
\def\@schapter#1{\if@twocolumn
                   \@topnewpage[\@makeschapterhead{#1}]%
                 \else
                   \@makeschapterhead{#1}%
                   \@afterheading
                 \fi}
\def\@makeschapterhead#1{%
  \vspace*{\prechapterskip}%
  {\parindent \z@
    \normalfont\chapterformat
    \interlinepenalty\@M
    \chaptertitleformat{#1}\par\nobreak
    \vskip \postchapterskip
  }}
\newcommand*\l@chapter[2]{%
  \ifnum \c@tocdepth >\m@ne
    \addpenalty{-\@highpenalty}%
    \vskip 1.0em \@plus\p@
    \setlength\@tempdima{1.5em}%
    \begingroup
      \parindent \z@ \rightskip \@pnumwidth
      \parfillskip -\@pnumwidth
      \leavevmode \large\bfseries
      \advance\leftskip\@tempdima
      \hskip -\leftskip
      #1\nobreak\hfil \nobreak\hb@xt@\@pnumwidth{\hss #2}\par
      \penalty\@highpenalty
    \endgroup
  \fi}
\let\l@appendix\l@chapter
\newcommand\chaptername{Chapter}
\newcommand\frontmatter{%
  \cleardoublepage
  \@mainmatterfalse
  \renewcommand*{\theHchapter}{front-\thechapter}%
  \pagenumbering{roman}%
  \morefrontmatter
}
\newcommand\morefrontmatter{}
\newcommand\mainmatter{%
  \cleardoublepage
  \@mainmattertrue
  \setcounter{chapter}{0}%
  \renewcommand*{\theHchapter}{\thechapter}%
  \pagenumbering{arabic}%
  \moremainmatter
}
\newcommand\moremainmatter{}
\newcommand\backmatter{%
  \if@openright
    \cleardoublepage
  \else
    \clearpage
  \fi
  \@mainmatterfalse}
\newcommand*{\booktocpreamble}{}
\newcommand\booktableofcontents{%
  \if@twocolumn
    \@restonecoltrue\onecolumn
  \else
    \@restonecolfalse
  \fi
  \chapter*{\contentsname
    \@mkboth{\MakeUppercase\contentsname}{\MakeUppercase\contentsname}}%
  \booktocpreamble
  \@starttoc{toc}%
  \if@restonecol\twocolumn\fi
}
\let\arttableofcontents\tableofcontents
\newcommand{\artpart}{%
  \def\toclevel@part{0}%
  \if@noskipsec \leavevmode\fi
  \par
  \addvspace{4ex}%
  \@afterindentfalse
  \secdef\@artpart\@sartpart
}
\let\@artpart\@part
\let\@sartpart\@spart
\newcommand\bookpart{%
  \def\toclevel@part{-1}%
  \if@openright
    \cleardoublepage
  \else
    \clearpage
  \fi
  \thispagestyle{plain}%
  \if@twocolumn
    \onecolumn
    \@tempswatrue
  \else
    \@tempswafalse
  \fi
  \preparthook
  \secdef\@bookpart\@sbookpart}
\newcommand{\parttitleformat}[1]{%
  \Huge\bfseries#1%
}
\newcommand*{\@parttoclabel}{part}
\def\@partapp{\partname}
\newcommand{\partnumberformat}[1]{%
  \Huge\bfseries \@partapp\nobreakspace#1\par\nobreak
  \vskip 20\p@
}
\newcommand{\preparthook}{\null\vfil}
\newcommand*{\partformat}{\centering}
\def\@bookpart[#1]#2{%
    \ifnum \c@secnumdepth >-2\relax
      \refstepcounter{part}%
      \addcontentsline{toc}{\@parttoclabel}{\protect\numberline{\thepart}#1}%
    \else
      \addcontentsline{toc}{\@parttoclabel}{#1}%
    \fi
    \markboth{}{}%
    {\interlinepenalty \@M
     \normalfont\partformat
     \ifnum \c@secnumdepth >-2\relax
       \partnumberformat{\thepart}%
     \fi
     \parttitleformat{#2}\par}%
    \postparthook}
\def\@sbookpart#1{%
    {\interlinepenalty \@M
     \normalfont\partformat
     \parttitleformat{#1}\par}%
    \postparthook}
\def\postparthook{\vfil\newpage
              \if@twoside
               \if@openright
                \null
                \thispagestyle{empty}%
                \newpage
               \fi
              \fi
              \if@tempswa
                \twocolumn
              \fi}
\newcommand\bookappendix{\par
  \setcounter{table}{0}%
  \setcounter{figure}{0}%
  \zeroextracounters
  \par
  \gdef\theHchapter{\Alph {chapter}}%
  \xdef\Hy@chapapp{\Hy@appendixstring}%
  \setcounter{chapter}{0}%
  \setcounter{section}{0}%
  \gdef\@chapapp{\appendixname}%
  \gdef\thechapter{\@Alph\c@chapter}%
  \csname appendixmore\endcsname
}
\newcommand*{\jmlrbookcommands}{%
  \let\part\bookpart
  \let\chapter\bookchapter
  \let\appendix\bookappendix
  \let\tableofcontents\booktableofcontents
  \def\thesection{\thechapter.\arabic{section}}%
}
\newcommand*{\jmlrarticlecommands}{%
  \let\part\artpart
  \let\chapter\artchapter
  \let\appendix\artappendix
  \let\tableofcontents\arttableofcontents
  \def\thesection{\arabic{section}}%
}
\newcommand*{\@jmlr@check@packages}{%
  \@ifpackageloaded{epsfig}{%
    \ClassError{jmlr}{Obsolete package `epsfig' detected.
     \MessageBreak
     Please use \string\includegraphics\space to include images
     instead}{}}{}%
  \@ifpackageloaded{psfig}{%
    \ClassError{jmlr}{Obsolete package `psfig' detected.
     \MessageBreak
     Please use \string\includegraphics\space to include images
     instead}{}}{}%
  \@ifpackageloaded{subfig}{%
    \ClassError{jmlr}{Package `subfig' detected.\MessageBreak
    This will cause a conflict if the article is incorporated
    \MessageBreak
    into a book using jmlbook.cls.
    \MessageBreak
    Please use \string\subfigure\space and
    \string\subtable\space instead}{}}{}%
  \@ifpackageloaded{theorem}{%
   \ClassError{jmlr}{Package `theorem' detected.\MessageBreak
    This can cause a conflict with other packages used by jmlr}{}}{}%
  \@ifpackageloaded{pdfpages}{Package `pdfpages' detected.\MessageBreak
   This can cause a problem for jmlrbook.}{}%
  \@ifpackageloaded{geometry}{Package `geometry' detected.\MessageBreak
   This can cause a problem for jmlrbook.}{}%
}
\AtBeginDocument{%
\@jmlr@check@packages
\let\@jmlr@check@packages\relax
}
\newcommand*{\jmlrSuppressPackageChecks}{%
  \let\@jmlr@check@packages\relax
}
\endinput
%%
%% End of file `jmlr.cls'.
