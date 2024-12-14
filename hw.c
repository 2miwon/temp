#include <linux/module.h>
#include <linux/proc_fs.h>
#include <linux/seq_file.h>

MODULE_AUTHOR("Heewon Lim");
MODULE_DESCRIPTION("System Programming 2024 - 2019147503");

#define HW_DIR "hw"
#define PROC_NAME "hw"
#define SCHEDULER_NAME "scheduler"
#define MEMORY_NAME "memory"

// 디렉토리와 파일 관련 변수
static struct proc_dir_entry *hw_dir;
static struct proc_dir_entry *scheduler_dir;
static struct proc_dir_entry *memory_dir;

// static int scheduler_show(struct seq_file *m, void *v);
// static int memory_show(struct seq_file *m, void *v);
// static int pid_info_open(struct inode *inode, struct file *file);
// static int pid_info_release(struct inode *inode, struct file *file);

// // 프로시저 파일 구조체
// static const struct file_operations scheduler_fops = {
//     .owner = THIS_MODULE,
//     .open = pid_info_open,
//     .read = seq_read,
//     .llseek = seq_lseek,
//     .release = pid_info_release,
// };

// static const struct file_operations memory_fops = {
//     .owner = THIS_MODULE,
//     .open = pid_info_open,
//     .read = seq_read,
//     .llseek = seq_lseek,
//     .release = pid_info_release,
// };

// 파일에서 PID를 읽어오는 함수
// static int pid_info_show(struct seq_file *m, void *v, const char *type)
// {
//     pid_t pid;
//     struct task_struct *task;
    
//     // PID 읽기
//     pid = simple_strtol((char *)v, NULL, 10);
//     task = pid_task(find_vpid(pid), PIDTYPE_PID);
    
//     if (!task) {
//         seq_printf(m, "Invalid PID\n");
//         return 0;
//     }
    
//     if (strcmp(type, "scheduler") == 0) {
//         // 스케줄러 관련 정보 출력 예시 (간단한 스케줄러 정보 출력)
//         seq_printf(m, "PID: %d, State: %ld\n", pid, task->state);
//         // 실제로는 여기서 스케줄러 정보 등을 출력해야 함
//     } else if (strcmp(type, "memory") == 0) {
//         // 메모리 관련 정보 출력 예시 (간단한 메모리 정보 출력)
//         seq_printf(m, "PID: %d, Memory: %ld kB\n", pid, task->mm->total_vm * 4);  // 예시로 메모리 사용량 출력
//     }

//     return 0;
// }

// static int pid_info_open(struct inode *inode, struct file *file)
// {
//     return single_open(file, pid_info_show, PDE_DATA(inode));
// }

// static int pid_info_release(struct inode *inode, struct file *file)
// {
//     return single_release(inode, file);
// }

static int __init hw_init(void) {
    // hw 디렉토리 생성
    hw_dir = proc_mkdir(HW_DIR, NULL);
    if (!hw_dir) {
        pr_err("Failed to create /proc/%s directory\n", HW_DIR);
        return -ENOMEM;
    }

    // scheduler 디렉토리 생성
    scheduler_dir = proc_mkdir(SCHEDULER_NAME, hw_dir);
    if (!scheduler_dir) {
        pr_err("Failed to create /proc/%s/%s directory\n", HW_DIR, SCHEDULER_NAME);
        return -ENOMEM;
    }

    // memory 디렉토리 생성
    memory_dir = proc_mkdir(MEMORY_NAME, hw_dir);
    if (!memory_dir) {
        pr_err("Failed to create /proc/%s/%s directory\n", HW_DIR, MEMORY_NAME);
        return -ENOMEM;
    }

    // PID 관련 파일 생성 (scheduler와 memory에 대해 각각)
    // if (!proc_create_data(SCHEDULER_NAME, 0, NULL, "scheduler")) {
    //     pr_err("Failed to create /proc/%s/%s/1234 file\n", HW_DIR, &scheduler_fops);
    //     return -ENOMEM;
    // }

    // if (!proc_create_data(MEMORY_NAME, 0, NULL, "memory")) {
    //     pr_err("Failed to create /proc/%s/%s/1234 file\n", HW_DIR, MEMORY_NAME);
    //     return -ENOMEM;
    // }

    return 0; 
}

static void __exit hw_exit(void) {
    // /proc 디렉토리와 파일 삭제
    remove_proc_entry(SCHEDULER_NAME, scheduler_dir);
    remove_proc_entry(MEMORY_NAME, memory_dir);
    remove_proc_entry(SCHEDULER_NAME, hw_dir);
    remove_proc_entry(MEMORY_NAME, hw_dir);
    remove_proc_entry(HW_DIR, NULL);

    pr_info("/proc/%s/%s and /proc/%s/%s and /proc/%s removed\n", HW_DIR, SCHEDULER_NAME, HW_DIR, MEMORY_NAME, HW_DIR);
}

module_init(hw_init);
module_exit(hw_exit);