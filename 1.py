# 用turtle库写出林字
import turtle

def draw_lin():
    # 初始化设置
    turtle.reset()
    turtle.speed(3)  # 设置绘制速度适中
    turtle.pensize(5)  # 设置笔画粗细
    
    # 绘制左边的"木"字
    turtle.penup()
    turtle.goto(-30, 0)
    turtle.pendown()
    
    # 绘制左边竖线
    turtle.setheading(90)  # 确保方向正确
    turtle.forward(100)
    
    # 绘制右边的"木"字
    turtle.penup()
    turtle.goto(30, 0)  # 调整位置，使两个"木"字之间有适当距离
    turtle.pendown()
    
    # 绘制右边竖线
    turtle.setheading(90)  # 确保方向正确
    turtle.forward(100)
    
    # 绘制左边"木"字的横线
    turtle.penup()
    turtle.goto(-55, 70)  # 调整起点，让横线比竖线长一些
    turtle.pendown()
    turtle.setheading(0)  # 设置方向为水平向右
    turtle.forward(50)
    
    # 绘制右边"木"字的横线
    turtle.penup()
    turtle.goto(5, 70)  # 调整起点，确保与左边对称
    turtle.pendown()
    turtle.setheading(0)  # 设置方向为水平向右
    turtle.forward(50)
    
    
    # 绘制左边"木"字横线下方的左侧撇
    turtle.penup()
    turtle.goto(-40, 50)  # 横线左侧位置
    turtle.pendown()
    turtle.setheading(245)  # 偏向左下方向
    turtle.forward(25)
    
    # 绘制左边"木"字横线下方的右侧撇
    turtle.penup()
    turtle.goto(-20, 50)  # 横线右侧位置
    turtle.pendown()
    turtle.setheading(315)  # 偏向右下方向
    turtle.forward(25)
    
    # 绘制右边"木"字横线下方的左侧撇
    turtle.penup()
    turtle.goto(20, 50)  # 横线左侧位置
    turtle.pendown()
    turtle.setheading(245)  # 偏向左下方向
    turtle.forward(25)
    
    # 绘制右边"木"字横线下方的右侧撇
    turtle.penup()
    turtle.goto(40, 50)  # 横线右侧位置
    turtle.pendown()
    turtle.setheading(315)  # 偏向右下方向，修正为与左边一致
    turtle.forward(25)

if __name__ == "__main__":
    turtle.title("绘制'林'字")  # 设置窗口标题
    draw_lin()
    turtle.hideturtle()  # 隐藏海龟
    turtle.done()  # 保持窗口显示