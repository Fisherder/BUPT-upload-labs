import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.InvalidPathException;
import java.nio.file.Path;
import java.util.Scanner;

/**
 * 文件内容统计工具
 * 功能：统计文件的行数、单词数和字符数
 */
public class FileContentAnalyzer {

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in, StandardCharsets.UTF_8);
        FileContentAnalyzer analyzer = new FileContentAnalyzer();

        System.out.println("欢迎使用文件内容统计工具");
        System.out.println("请输入文件路径（输入 'exit' 退出程序）：");

        while (true) {
            String inputPath = scanner.nextLine().trim();
            if ("exit".equalsIgnoreCase(inputPath)) {
                System.out.println("程序已退出。");
                break;
            }

            try {
                FileStats stats = analyzer.analyzeFile(Path.of(inputPath));
                System.out.printf(
                    "统计结果：\n文件路径: %s\n行数: %d\n单词数: %d\n字符数: %d%n%n",
                    inputPath, stats.getLines(), stats.getWords(), stats.getChars()
                );
            } catch (InvalidPathException e) {
                System.err.println("错误: 路径格式不正确 - " + e.getMessage());
            } catch (IOException e) {
                System.err.println("错误: 文件读取失败 - " + e.getMessage());
            } catch (IllegalArgumentException e) {
                System.err.println("错误: " + e.getMessage());
            }

            System.out.println("请输入下一个文件路径（输入 'exit' 退出程序）：");
        }

        scanner.close();
    }

    /**
     * 分析文件内容并返回统计结果
     *
     * @param filePath 文件路径
     * @return 包含行数、单词数和字符数的 FileStats 对象
     * @throws IOException              如果文件不存在或无法读取
     * @throws IllegalArgumentException 如果路径不是文件
     */
    public FileStats analyzeFile(Path filePath) throws IOException {
        if (!Files.exists(filePath)) {
            throw new IllegalArgumentException("文件不存在: " + filePath);
        }
        if (!Files.isRegularFile(filePath)) {
            throw new IllegalArgumentException("路径不是一个普通文件: " + filePath);
        }

        int lines = 0, words = 0, chars = 0;
        // 直接使用 Files.lines 来读取每一行
        try (var stream = Files.lines(filePath, StandardCharsets.UTF_8)) {
            for (String line : (Iterable<String>) stream::iterator) {
                lines++;
                // 原来：
                // chars += line.length();
                // 新增：把系统行分隔符也算入字符数
                chars += line.length() + System.lineSeparator().length();
            
                String trimmed = line.trim();
                if (!trimmed.isEmpty()) {
                    words += trimmed.split("\\s+").length;
                }
            }
        }
        return new FileStats(lines, words, chars);
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
            return String.format("行数: %d, 单词数: %d, 字符数: %d", lines, words, chars);
        }
    }
}
