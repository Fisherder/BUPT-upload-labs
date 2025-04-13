import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.Scanner;

/**
 * 文件内容统计工具
 * 功能：统计文件的行数、单词数和字符数
 */
public class FileContentAnalyzer {

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        FileContentAnalyzer analyzer = new FileContentAnalyzer();

        System.out.println("欢迎使用文件内容统计工具");
        System.out.println("请输入文件路径（输入 'exit' 退出程序）：");

        while (true) {
            String inputPath = scanner.nextLine();
            if ("exit".equalsIgnoreCase(inputPath)) {
                System.out.println("程序已退出。");
                break;
            }

            try {
                FileStats stats = analyzer.analyzeFile(inputPath);
                System.out.println("\n统计结果：");
                System.out.println("文件路径: " + inputPath);
                System.out.println("行数: " + stats.getLines());
                System.out.println("单词数: " + stats.getWords());
                System.out.println("字符数: " + stats.getChars());
                System.out.println("\n请输入下一个文件路径（输入 'exit' 退出程序）：");
            } catch (Exception e) {
                System.out.println("错误: " + e.getMessage());
                System.out.println("请输入有效的文件路径：");
            }
        }

        scanner.close();
    }

    /**
     * 分析文件内容并返回统计结果
     *
     * @param filePath 文件路径
     * @return 包含行数、单词数和字符数的 FileStats 对象
     * @throws Exception 如果文件不存在或无法读取
     */
    public FileStats analyzeFile(String filePath) throws Exception {
        File file = new File(filePath);

        if (!file.exists()) {
            throw new Exception("文件不存在");
        }

        if (!file.isFile()) {
            throw new Exception("路径不是一个文件");
        }

        int lineCount = 0;
        int wordCount = 0;
        int charCount = 0;

        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(new FileInputStream(file), "UTF-8"))) {

            String line;
            while ((line = reader.readLine()) != null) {
                lineCount++;

                // 统计字符数（包括空格）
                charCount += line.length();

                // 统计单词数（以空格、制表符或换行符分隔）
                String[] words = line.split("\\s+");
                wordCount += words.length;
            }
        }

        return new FileStats(lineCount, wordCount, charCount);
    }

    /**
     * 文件统计结果类
     */
    public static class FileStats {
        private final int lines;
        private final int words;
        private final int chars;

        public FileStats(int lines, int words, int chars) {
            this.lines = lines;
            this.words = words;
            this.chars = chars;
        }

        public int getLines() {
            return lines;
        }

        public int getWords() {
            return words;
        }

        public int getChars() {
            return chars;
        }

        @Override
        public String toString() {
            return "行数: " + lines + ", 单词数: " + words + ", 字符数: " + chars;
        }
    }
}