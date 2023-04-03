use 5.036000;
use ExtUtils::MakeMaker;
use ExtUtils::PkgConfig;
use FindBin;

my @cflags = ('-Icpp/src');
my @libs;
for my $dep (qw(eigen3 assimp dart)) {
    if (ExtUtils::PkgConfig->exists($dep)) {
	my $cflag = ExtUtils::PkgConfig->cflags($dep);
	my $lib = ExtUtils::PkgConfig->libs($dep);
	push @cflags, $cflag if $cflag;
	push @libs, $lib if $lib;
    } else {
	push @libs, "-l$dep";
    }
}

sub MY::postamble {
    return <<'MAKE_LIB';
$(MYEXTLIB): cpp/Makefile
	cd cpp && $(MAKE)

cpp/Makefile: cpp/CMakeLists.txt
	cd cpp && cmake -DCCCDLFLAGS=$(CCCDLFLAGS) .
MAKE_LIB
}

# See lib/ExtUtils/MakeMaker.pm for details of how to influence
# the contents of the Makefile that is written.
WriteMakefile(
    NAME              => 'CharacterEnv',
    VERSION_FROM      => 'lib/CharacterEnv.pm', # finds $VERSION, requires EU::MM from perl >= 5.5
    PREREQ_PM         => {}, # e.g., Module::Name => 1.1
    ABSTRACT_FROM     => 'lib/CharacterEnv.pm', # retrieve abstract from module
    AUTHOR            => 'Kaixiang Xie <kaixiang.xie@mail.mcgill.ca>',
    #LICENSE           => 'perl',
    #Value must be from legacy list of licenses here
    #https://metacpan.org/pod/Module::Build::API
    #LIBS              => ["-Wl,-rpath=$FindBin::Bin/cpp -L$FindBin::Bin/cpp -lCharacterEnv"], # e.g., '-lm'
    LIBS              => ["@libs"], # e.g., '-lm'
    DEFINE            => '', # e.g., '-DHAVE_SOMETHING'
    INC               => "@cflags", # e.g., '-I. -I/usr/include/other'
	# Un-comment this if you add C files to link with later:
    # OBJECT            => '$(O_FILES)', # link all the C files too
    CC                => 'g++',
    LD                => 'g++',
    MYEXTLIB          => 'cpp/libCharacterEnv$(LIB_EXT)',
    depend            => { 'CharacterEnv.o' => 'cpp/libCharacterEnv.a' },
    clean             => { FILES => 'cpp/libCharacterEnv.a' },
);
