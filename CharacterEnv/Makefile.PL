use 5.014000;
use ExtUtils::MakeMaker;
use ExtUtils::PkgConfig;
use FindBin;

# TODO:
#   fix
#	* dart.pc depends on eigen (should be eigen3)
#	* dart.pc depends on boost (not found)
#	* "incorrect" includedir in dart.pc: .../include or .../include/dart ?
#	* dart-12 uses C++14 but FCL-5 uses C++11
#	* maybe use cmake instead of pkg-config?

my @cflags = ('-std=c++14 -Icpp/src');
my @libs;
for my $dep (qw(eigen3 assimp dart)) {
    my $prefix = "\U${dep}_INSTALL_PREFIX";
    if (ExtUtils::PkgConfig->exists($dep)) {
	my $cflag = ExtUtils::PkgConfig->cflags($dep);
	my $lib = ExtUtils::PkgConfig->libs($dep);
	push @cflags, $cflag if $cflag;
	push @libs, $lib if $lib;
    } else {
        my $path = $ENV{$prefix};
        if (defined($path)) {
            push @cflags, "-I$path/include";
            push @libs, "-L$path/lib";
        } else {
            warn "Cannot find $dep. Please define $prefix\n";
        }
	push @libs, "-l$dep";
    }
}

sub MY::postamble {
    return <<'MAKE_LIB';
$(MYEXTLIB): cpp/Makefile cpp/src/*.h cpp/src/*.cpp
	cd cpp && $(MAKE)

cpp/Makefile: cpp/CMakeLists.txt
	cd cpp && cmake -DCMAKE_CXX_STANDARD=14 -DCMAKE_CXX_FLAGS=$(CCCDLFLAGS) -DCMAKE_BUILD_TYPE=release .

cleancpp: 
	cd cpp && $(Make) clean
MAKE_LIB
}

# See lib/ExtUtils/MakeMaker.pm for details of how to influence
# the contents of the Makefile that is written.
WriteMakefile(
    NAME              => 'CharacterEnv',
    VERSION_FROM      => 'lib/CharacterEnv.pm', # finds $VERSION, requires EU::MM from perl >= 5.5
    CONFIGURE_REQUIRES=> { 'ExtUtils::PkgConfig' => 0 },
    PREREQ_PM         => {
	'AI::MXNet' => 0,
    }, # e.g., Module::Name => 1.1
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
    OPTIMIZE          => '-O3',
    MYEXTLIB          => 'cpp/libCharacterEnv$(LIB_EXT)',
    depend            => { 'CharacterEnv.o' => 'cpp/libCharacterEnv.a' },
    clean             => { FILES => 'cpp/libCharacterEnv.a cpp/Makefile cpp/CMakeCache.txt cpp/CMakeFiles' },
);
